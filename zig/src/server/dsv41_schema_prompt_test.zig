//! DeepSeek-V4.1's ``## Response Format`` block (TF_DSV41_SCHEMA_PROMPT): the deployed chat template renders it as the
//! reference encoder (encoding/encoding.py, render_message of a system message with a response_format) does, and with
//! the switch off the prompt ids are byte for byte what they were before the block existed.
//!
//! fixtures/dsv41_response_format: cases.json is the response_format records (no images) of DeepSeek's recorded encoder
//! outputs (TensorFold's zig/src/server/fixtures/deepseek_v41_encoder/cases.jsonl.gz, MIT, LICENSE beside it), their
//! fields unchanged; chat_template_c282975e.jinja is the deployed template before the block (sha256 c282975e...).
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
const Value = json.Value;

const cases_json = @embedFile("fixtures/dsv41_response_format/cases.json");
const template_before = @embedFile("fixtures/dsv41_response_format/chat_template_c282975e.jinja");
const block_head = "## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n";

/// An HfText that holds only a compiled template: its own render (the server's Jinja context), bytes for tokens.
const Rig = struct {
    hf: HfText,
    vt: model_text.Text.VTable,

    fn init(a: std.mem.Allocator, source: []const u8) !Rig {
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

    fn text(r: *Rig) model_text.Text {
        return .{ .ctx = &r.hf, .vtable = &r.vt };
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const out = try a.alloc(u8, ids.len);
        for (ids, out) |id, *byte| byte.* = @intCast(id & 0xff);
        return out;
    }

    fn tokenId(_: *anyopaque, _: []const u8) ?u32 {
        return null;
    }

    fn tokenString(_: *anyopaque, a: std.mem.Allocator, id: u32) std.mem.Allocator.Error![]u8 {
        return std.fmt.allocPrint(a, "{d}", .{id});
    }

    fn vocabSize(_: *anyopaque) u32 {
        return 256;
    }

    fn eosIds(_: *anyopaque) []const u32 {
        return &.{};
    }
};

/// One recorded case's render options; `rf` the OpenAI response_format to show, or null.
fn options(a: std.mem.Allocator, c: Value, rf: ?Value) !model_text.RenderOptions {
    const thinking = c.get("thinking").?.bool;
    const tools = c.get("tools").?;
    return .{
        .tools = if (tools == .array and tools.array.len > 0) tools else null,
        .add_generation_prompt = c.get("generation").?.bool,
        .enable_thinking = thinking,
        .reasoning_effort_value = if (thinking) c.get("budget").?.int64() else null,
        .response_format = if (rf) |f| try deepseek_schema.schemaFor(a, f) else null,
    };
}

fn render(a: std.mem.Allocator, rig: *Rig, c: Value, rf: ?Value) ?[]u8 {
    var problem: []const u8 = "";
    const o = options(a, c, rf) catch return null;
    return rig.text().render(a, c.get("messages").?, o, &problem) catch null;
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

/// The records whose whole recorded prompt the deployed template reproduces. The others differ elsewhere, where the
/// server normalizes a request before rendering (tool-call arguments as JSON text, content parts the chat route turns
/// into text, a tool's missing description, drop_thinking=False, a budget the recording did not apply); their block is
/// checked on its own.
const whole = [_]i64{ 41, 47, 78, 119, 190, 210, 217, 230, 235, 261, 275 };

/// The text after the block's head up to the next special token: the schema as the encoder wrote it.
fn schemaText(prompt_text: []const u8) ?[]const u8 {
    const at = (std.mem.indexOf(u8, prompt_text, block_head) orelse return null) + block_head.len;
    const end = std.mem.indexOfPos(u8, prompt_text, at, "<\u{ff5c}") orelse prompt_text.len;
    return prompt_text[at..end];
}

test "the deployed template renders the encoder's ## Response Format block on DeepSeek's recorded prompts" {
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
        // the block is all the response_format adds: "\n\n" + the head + the encoder's schema text at the end of the
        // first system message, which is the empty one the encoder's caller adds when the conversation has none
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

test "with no response_format the template renders every recorded conversation as the template before the block did" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var now = try Rig.init(a, assets.chat_template);
    var before = try Rig.init(a, template_before);
    var rendered: usize = 0;
    for (try cases(a)) |c| {
        const x = render(a, &now, c, null);
        const y = render(a, &before, c, null);
        try std.testing.expectEqual(x == null, y == null);
        if (x) |text| {
            try std.testing.expectEqualStrings(y.?, text);
            rendered += 1;
        }
    }
    try std.testing.expect(rendered >= 60);
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

/// A chat request's prompt ids through prompt.prepare (the chat route's), on a server over `source` with the switch.
fn prepared(a: std.mem.Allocator, source: []const u8, on: bool, body: []const u8) ![]const u32 {
    var rig = try Rig.init(a, source);
    var idle: Idle = .{};
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, idle.engine(), rig.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false, .dsv41_schema_prompt = on }, null);
    defer srv.deinit();
    try std.testing.expectEqual(@import("messages.zig").Dialect.deepseek_v41, srv.dialect);
    const raw = (try json.parse(a, body)).ok;
    const fields = try json.newObject(a);
    if (raw.get("response_format")) |rf| try fields.put(a, "response_format", rf);
    var cx: errors.Cx = .{ .a = a };
    const input: chat.Input = .{ .messages = raw.get("messages").?, .fields = .{ .object = fields } };
    return (try prompt.prepare(srv, &cx, input, false, null)).ids;
}

test "TF_DSV41_SCHEMA_PROMPT off: a request's response_format leaves its prompt ids as the template before the block made them" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const messages = "\"messages\":[{\"role\":\"user\",\"content\":\"Give me a number.\"}]";
    const plain = try std.fmt.allocPrint(a, "{{{s}}}", .{messages});
    const formats = [_][]const u8{
        "{\"type\":\"json_object\"}",
        "{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"n\",\"schema\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"number\"}}}}}",
    };
    const before = try prepared(a, template_before, false, plain);
    for (formats) |f| {
        const body = try std.fmt.allocPrint(a, "{{{s},\"response_format\":{s}}}", .{ messages, f });
        // off: byte for byte the ids the template before the block gave, with or without the response_format
        try std.testing.expectEqualSlices(u32, before, try prepared(a, assets.chat_template, false, body));
        try std.testing.expectEqualSlices(u32, before, try prepared(a, assets.chat_template, false, plain));
        // on: the block, on an empty system message the encoder's way
        const on = try prepared(a, assets.chat_template, true, body);
        const shown = try std.mem.concat(a, u8, &.{ "<\u{ff5c}System\u{ff5c}>\n\n", block_head });
        const bytes = try a.alloc(u8, on.len);
        for (on, bytes) |id, *b| b.* = @intCast(id);
        try std.testing.expect(std.mem.indexOf(u8, bytes, shown) != null);
        // on, without a response_format: as before
        try std.testing.expectEqualSlices(u32, before, try prepared(a, assets.chat_template, true, plain));
    }
}
