//! The ``## Response Format`` block as encoding.py renders it (deepseek_schema.render); with it off, the ids as before.
const std = @import("std");
const api = @import("engine_api");
const template = @import("template");
const assets = @import("dsv41_assets");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const HfText = @import("hf_text.zig").HfText;
const prompt = @import("prompt.zig");
const chat = @import("chat.zig");
const deepseek_schema = @import("deepseek_schema.zig");
const errors = @import("errors.zig");
const messages_mod = @import("messages.zig");
const Value = json.Value;

const cases_json = @embedFile("fixtures/dsv41_response_format/cases.json"); // from TensorFold, MIT (LICENSE)
const block_head = "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n";

/// An HfText holding only a compiled template: its own render, bytes for tokens, the image placeholder as `image_id`.
pub const Rig = struct {
    hf: HfText,
    vt: model_text.Text.VTable,

    pub fn init(a: std.mem.Allocator, source: []const u8) !Rig {
        var diag: template.Diag = .{};
        var r: Rig = .{ .hf = .{ .gpa = a, .arena = .init(a), .tok = undefined, .source = source, .compiled = try template.compile(a, source, &diag) }, .vt = undefined };
        const base = r.hf.text().vtable.*;
        r.vt = base;
        r.vt.encode = encode;
        r.vt.decode = decode;
        r.vt.token_id = tokenId;
        r.vt.token_string = tokenString;
        r.vt.vocab_size = vocabSize;
        r.vt.eos_ids = eosIds;
        r.vt.special = null;
        return r;
    }

    pub const image_placeholder = "<\u{ff5c}deepseek_image\u{ff5c}>";
    pub const image_id: u32 = 300;

    pub fn text(r: *Rig) model_text.Text {
        return .{ .ctx = &r.hf, .vtable = &r.vt };
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        var ids: std.ArrayList(u32) = .empty;
        var i: usize = 0;
        while (i < input.len) {
            if (std.mem.startsWith(u8, input[i..], image_placeholder)) {
                try ids.append(a, image_id);
                i += image_placeholder.len;
            } else {
                try ids.append(a, input[i]);
                i += 1;
            }
        }
        return ids.items;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        for (ids) |id| if (id == image_id) try out.appendSlice(a, image_placeholder) else try out.append(a, @intCast(id & 0xff));
        return out.items;
    }

    fn tokenId(_: *anyopaque, piece: []const u8) ?u32 {
        return if (std.mem.eql(u8, piece, image_placeholder)) image_id else null;
    }

    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 512;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
};

/// One recorded case's render options.
fn options(c: Value) model_text.RenderOptions {
    const thinking = c.get("thinking").?.bool;
    const tools = c.get("tools").?;
    return .{
        .tools = if (tools == .array and tools.array.len > 0) tools else null,
        .add_generation_prompt = c.get("generation").?.bool,
        .enable_thinking = thinking,
        .reasoning_effort_value = if (thinking) c.get("budget").?.int64() else null,
    };
}

/// A case rendered with `rf` (an OpenAI response_format) through deepseek_schema.render, or the template's own without.
fn render(a: std.mem.Allocator, rig: *Rig, c: Value, rf: ?Value) ?[]u8 {
    var problem: []const u8 = "";
    const schema = if (rf) |f| (deepseek_schema.schemaFor(a, f) catch return null) else null;
    if (schema) |v| return deepseek_schema.render(rig.text(), a, c.get("messages").?, options(c), v, &problem) catch null;
    return rig.text().render(a, c.get("messages").?, options(c), &problem) catch null;
}

/// ``{"type": "json_schema", "json_schema": {"schema": schema}}``.
fn jsonSchema(a: std.mem.Allocator, schema: Value) !Value {
    const js = try json.newObject(a);
    try js.put(a, "schema", schema);
    const rf = try json.newObject(a);
    try rf.put(a, "type", .{ .string = "json_schema" });
    try rf.put(a, "json_schema", .{ .object = js });
    return .{ .object = rf };
}

/// The recorded prompt with its schema (the text after the block's head, up to the next special token) replaced.
fn withSchema(a: std.mem.Allocator, recorded: []const u8, schema: []const u8) ![]u8 {
    const at = (std.mem.indexOf(u8, recorded, block_head) orelse return error.NoBlock) + block_head.len;
    const end = std.mem.indexOfPos(u8, recorded, at, "<\u{ff5c}") orelse recorded.len;
    return std.mem.concat(a, u8, &.{ recorded[0..at], schema, recorded[end..] });
}

fn cases(a: std.mem.Allocator) ![]Value {
    return (try json.parse(a, cases_json)).ok.array;
}

/// The records whose whole prompt the template reproduces (the others differ where the server normalizes a request).
const whole = [_]i64{ 41, 47, 78, 119, 190, 210, 217, 230, 235, 261, 275 };

/// The text after the block's head up to the next special token: the schema as the encoder wrote it.
fn schemaText(prompt_text: []const u8) ?[]const u8 {
    const at = (std.mem.indexOf(u8, prompt_text, block_head) orelse return null) + block_head.len;
    const end = std.mem.indexOfPos(u8, prompt_text, at, "<\u{ff5c}") orelse prompt_text.len;
    return prompt_text[at..end];
}

test "the server adds the encoder's ## Response Format block to DeepSeek's recorded prompts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig = try Rig.init(a, assets.chat_template);
    var exact: usize = 0;
    var checked: usize = 0;
    for (try cases(a)) |c| {
        const recorded = c.get("prompt").?.string;
        const id = c.get("id").?.int64().?;
        const schema = schemaText(recorded) orelse return error.NoRecordedBlock;
        const with = render(a, &rig, c, try jsonSchema(a, c.get("response_format").?)) orelse return error.Render;
        const without = render(a, &rig, c, null) orelse return error.Render;
        // all it adds: "\n\n", the head and the schema text, ending the first system message (an empty one if none)
        const block = try std.mem.concat(a, u8, &.{ "\n\n", block_head, schema });
        const at = std.mem.indexOf(u8, with, block) orelse {
            std.debug.print("case {d}: no block like the recording's\n", .{id});
            return error.BlockDiffers;
        };
        const added_system = with.len == without.len + block.len + "<\u{ff5c}System\u{ff5c}>".len;
        const start = if (added_system) at - "<\u{ff5c}System\u{ff5c}>".len else at;
        if (added_system) try std.testing.expectEqualStrings("<\u{ff5c}System\u{ff5c}>", with[start..at]);
        try std.testing.expectEqualStrings(without, try std.mem.concat(a, u8, &.{ with[0..start], with[at + block.len ..] }));
        // and the recording has the same text before the block in its system message: the block closes it there too
        try std.testing.expect(std.mem.indexOf(u8, recorded, with[start .. at + block.len]) != null);
        checked += 1;
        if (std.mem.indexOfScalar(i64, &whole, id) != null) {
            try std.testing.expectEqualStrings(recorded, with);
            exact += 1;
        }
        // json_object: the same prompt with {"type": "object"} as the schema, as schema_for_prompt maps it
        const obj = try json.newObject(a);
        try obj.put(a, "type", .{ .string = "json_object" });
        const as_object = render(a, &rig, c, .{ .object = obj }) orelse return error.Render;
        try std.testing.expectEqualStrings(try withSchema(a, with, "{\"type\": \"object\"}"), as_object);
    }
    try std.testing.expectEqual(@as(usize, 69), checked);
    try std.testing.expectEqual(whole.len, exact);
}

test "the shipped chat template is the pinned c282975e: the block comes from the server, not the template" {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(assets.chat_template, &digest, .{});
    try std.testing.expectEqualStrings("c282975e81949382a878e06060d1ef3e6e7b22628f9b51ee2188b2dfbcc7e217", &std.fmt.bytesToHex(digest, .lower));
}

/// Answers nothing: these tests only prepare prompts.
const Idle = struct {
    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }
    fn info(_: *anyopaque) api.Info {
        return .{};
    }
    fn submit(_: *anyopaque, _: api.Id, _: *const api.Request, _: api.Sink) api.SubmitError!void {
        return error.Busy;
    }
    fn cancel(_: *anyopaque, _: api.Id) void {}
    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }
    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// A server over the shipped template (`on`: TF_DSV41_SCHEMA_PROMPT=1) and its Rig.
const Served = struct {
    rig: Rig,
    idle: Idle = .{},
    srv: *server.Server = undefined,

    fn start(s: *Served, a: std.mem.Allocator, on: bool) !void {
        s.rig = try Rig.init(a, assets.chat_template);
        s.idle = .{};
        s.srv = try server.Server.init(std.testing.allocator, std.testing.io, s.idle.engine(), s.rig.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false, .dsv41_schema_prompt = on }, null);
        try std.testing.expectEqual(messages_mod.Dialect.deepseek_v41, s.srv.dialect);
    }

    /// A chat request's prompt ids through prompt.prepare (the chat route's); `rf` its response_format, or null.
    fn prepared(s: *Served, a: std.mem.Allocator, messages: Value, tools: []const Value, thinking: bool, rf: ?Value) ![]const u32 {
        const fields = try json.newObject(a);
        if (rf) |f| try fields.put(a, "response_format", f);
        var cx: errors.Cx = .{ .a = a };
        const input: chat.Input = .{ .messages = messages, .tools = tools, .fields = .{ .object = fields } };
        return (try prompt.prepare(s.srv, &cx, input, thinking, null)).ids;
    }
};

/// The ids as the bytes they stand for (the Rig's tokens are bytes).
fn bytesOf(a: std.mem.Allocator, ids: []const u32) ![]u8 {
    const out = try a.alloc(u8, ids.len);
    for (ids, out) |id, *b| b.* = @intCast(id);
    return out;
}

test "TF_DSV41_SCHEMA_PROMPT off: every recorded conversation's prompt ids are the template's, with or without its schema" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var off: Served = undefined;
    try off.start(a, false);
    defer off.srv.deinit();
    var on: Served = undefined;
    try on.start(a, true);
    defer on.srv.deinit();
    var compared: usize = 0;
    for (try cases(a)) |c| {
        const tools: []const Value = if (c.get("tools").? == .array) c.get("tools").?.array else &.{};
        const thinking = c.get("thinking").?.bool;
        const rf = try jsonSchema(a, c.get("response_format").?);
        // the server refuses some recorded messages (content parts the encoder takes, the image marker as text)
        const plain = off.prepared(a, c.get("messages").?, tools, thinking, null) catch |e| if (e == error.Refused) continue else return e;
        try std.testing.expectEqualSlices(u32, plain, try off.prepared(a, c.get("messages").?, tools, thinking, rf));
        try std.testing.expectEqualSlices(u32, plain, try on.prepared(a, c.get("messages").?, tools, thinking, null));
        // and they are the template's own render of the server's normalized messages, as before the switch existed
        var o = options(c);
        o.add_generation_prompt = true;
        o.reasoning_effort_value = null;
        var cx: errors.Cx = .{ .a = a };
        const srv = off.srv;
        const normalized = try messages_mod.normalize(&cx, c.get("messages").?, srv.late_system, srv.needs_user_after_tool, srv.dialect);
        const ready = try messages_mod.toolArguments(&cx, normalized, srv.dialect);
        var problem: []const u8 = "";
        try std.testing.expectEqualSlices(u32, try off.rig.text().renderIds(a, ready, o, &problem), plain);
        compared += 1;
    }
    try std.testing.expectEqual(@as(usize, 32), compared);
}

test "the switch on adds the block only to a request with a response_format, on an empty system message the encoder's way" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var off: Served = undefined;
    try off.start(a, false);
    defer off.srv.deinit();
    var on: Served = undefined;
    try on.start(a, true);
    defer on.srv.deinit();
    const messages = (try json.parse(a, "[{\"role\":\"user\",\"content\":\"Give me a number.\"}]")).ok;
    const formats = [_][]const u8{
        "{\"type\":\"json_object\"}",
        "{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"n\",\"schema\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"number\"}}}}}",
    };
    const plain = try off.prepared(a, messages, &.{}, false, null);
    try std.testing.expectEqualSlices(u32, plain, try on.prepared(a, messages, &.{}, false, null));
    for (formats) |f| {
        const rf = (try json.parse(a, f)).ok;
        try std.testing.expectEqualSlices(u32, plain, try off.prepared(a, messages, &.{}, false, rf));
        const with = try bytesOf(a, try on.prepared(a, messages, &.{}, false, rf));
        const shown = try std.mem.concat(a, u8, &.{ "<\u{ff5c}System\u{ff5c}>\n\n", block_head });
        const at = std.mem.indexOf(u8, with, shown) orelse return error.NoBlock;
        // without the empty system message and its block: the plain prompt
        const end = std.mem.indexOfPos(u8, with, at + shown.len, "<\u{ff5c}User\u{ff5c}>").?;
        try std.testing.expectEqualStrings(try bytesOf(a, plain), try std.mem.concat(a, u8, &.{ with[0..at], with[end..] }));
    }
}

test "a falsy schema ({} as an object or as text) leaves the prompt as without a response_format" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var on: Served = undefined;
    try on.start(a, true);
    defer on.srv.deinit();
    const messages = (try json.parse(a, "[{\"role\":\"user\",\"content\":\"Give me a number.\"}]")).ok;
    const plain = try on.prepared(a, messages, &.{}, false, null);
    for ([_][]const u8{ "{}", "\"{}\"" }) |schema| {
        const f = try std.fmt.allocPrint(a, "{{\"type\":\"json_schema\",\"json_schema\":{{\"name\":\"n\",\"schema\":{s}}}}}", .{schema});
        const rf = (try json.parse(a, f)).ok;
        try std.testing.expectEqual(@as(?Value, null), try deepseek_schema.schemaFor(a, rf));
        try std.testing.expectEqualSlices(u32, plain, try on.prepared(a, messages, &.{}, false, rf));
    }
}

test "an empty conversation with a schema: one empty system message carrying the block, as the encoder renders it" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig = try Rig.init(a, assets.chat_template);
    const schema = (try json.parse(a, "{\"type\": \"object\"}")).ok;
    var problem: []const u8 = "";
    const empty: Value = .{ .array = &.{} };
    const text = try deepseek_schema.render(rig.text(), a, empty, .{}, schema, &problem);
    const want = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}System\u{ff5c}>\n\n" ++ block_head ++ "{\"type\": \"object\"}";
    try std.testing.expectEqualStrings(want, text);
}

/// Renders a message count: a template whose render of the first message is no prefix of the whole conversation's.
fn countRender(_: *anyopaque, a: std.mem.Allocator, messages: Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
    return std.fmt.allocPrint(a, "<{d}>", .{messages.array.len});
}

test "a template whose system message alone is no prefix of the whole render: the block is refused, not misplaced" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var rig = try Rig.init(a, assets.chat_template);
    rig.vt.render = countRender;
    const messages = (try json.parse(a, "[{\"role\":\"user\",\"content\":\"hi\"}]")).ok;
    const schema = (try json.parse(a, "{\"type\": \"object\"}")).ok;
    var problem: []const u8 = "";
    try std.testing.expectError(error.Template, deepseek_schema.render(rig.text(), a, messages, .{}, schema, &problem));
    try std.testing.expect(problem.len > 0);
}
