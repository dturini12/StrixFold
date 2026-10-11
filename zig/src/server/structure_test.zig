//! Structured output through the chat handler: what reaches an engine that enforces structures (the request's
//! structure, a DeepSeek-V4.1 required call as a DSML calls grammar), and what is refused before any header (a schema
//! the grammar compiler does not take, an engine without structures).
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const errors = @import("errors.zig");
const Conn = @import("http_conn.zig").Conn;

/// Bytes as tokens; every prompt renders as "prompt". `dsml`: a template the DeepSeek-V4.1 dialect reads.
const TestText = struct {
    dsml: bool = false,

    fn text(t: *@This()) model_text.Text {
        return .{ .ctx = t, .vtable = &.{ .encode = encode, .decode = decode, .token_id = tokenId, .token_string = tokenString, .vocab_size = vocabSize, .eos_ids = eosIds, .render = render, .template_source = templateSource } };
    }

    fn encode(_: *anyopaque, a: std.mem.Allocator, input: []const u8, _: bool) model_text.Error![]u32 {
        const ids = try a.alloc(u32, input.len);
        for (input, ids) |byte, *id| id.* = byte;
        return ids;
    }

    fn decode(_: *anyopaque, a: std.mem.Allocator, ids: []const u32) std.mem.Allocator.Error![]u8 {
        const decoded = try a.alloc(u8, ids.len);
        for (ids, decoded) |id, *byte| byte.* = @intCast(id);
        return decoded;
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

    fn render(_: *anyopaque, a: std.mem.Allocator, _: json.Value, _: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        return a.dupe(u8, "prompt");
    }

    fn templateSource(ctx: *anyopaque) []const u8 {
        const t: *TestText = @ptrCast(@alignCast(ctx));
        return if (t.dsml) "{%- set REMINDER = '<\u{ff5c}latest_reminder\u{ff5c}>' -%} <\u{ff5c}DSML\u{ff5c} invoke name=\"$TOOL_NAME\">" else "";
    }
};

/// An engine that enforces structures (or not) and keeps the last request's structure; it answers "{}".
const Engine = struct {
    structures: bool = true,
    kind: ?@FieldType(api.Structure, "kind") = null,
    text: [512]u8 = undefined,
    len: usize = 0,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 64, .structures = e.structures };
    }

    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        if (r.structure) |s| {
            e.kind = s.kind;
            e.len = @min(s.text.len, e.text.len);
            @memcpy(e.text[0..e.len], s.text[0..e.len]);
        }
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ '{', '}' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// A whole reply's status and error message.
const Seen = struct {
    status: ?u16 = null,
    message: [512]u8 = undefined,
    len: usize = 0,

    fn out(c: *Seen) openai.Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(_: *anyopaque, _: usize) error{Closed}!void {}

    fn event(_: *anyopaque, _: ?json.Value) error{Closed}!void {}

    fn reply(ctx: *anyopaque, status: u16, payload: json.Value) void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.status = status;
        const m = (payload.get("error") orelse return).get("message") orelse return;
        if (m != .string) return;
        c.len = @min(m.string.len, c.message.len);
        @memcpy(c.message[0..c.len], m.string[0..c.len]);
    }

    fn text(c: *const Seen) []const u8 {
        return c.message[0..c.len];
    }
};

fn send(e: *Engine, dsml: bool, body: []const u8) !Seen {
    var text: TestText = .{ .dsml = dsml };
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, e.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
    defer srv.deinit();
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen: Seen = .{};
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
    return seen;
}

const user = "\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":8";

test "a schema reaches the engine that enforces it" {
    var e: Engine = .{};
    const seen = try send(&e, false, "{" ++ user ++ ",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"p\",\"schema\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}}}}}}");
    try std.testing.expectEqual(@as(?u16, 200), seen.status);
    try std.testing.expectEqual(.json_schema, e.kind.?);
    try std.testing.expect(std.mem.indexOf(u8, e.text[0..e.len], "\"integer\"") != null);
}

test "a schema keyword the grammar cannot enforce is refused before any header, in the compiler's words" {
    var e: Engine = .{};
    const seen = try send(&e, false, "{" ++ user ++ ",\"response_format\":{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"p\",\"schema\":{\"type\":\"array\",\"uniqueItems\":true}}}}");
    try std.testing.expectEqual(@as(?u16, 400), seen.status);
    try std.testing.expect(std.mem.indexOf(u8, seen.text(), "uniqueItems") != null);
    try std.testing.expectEqual(@as(?@FieldType(api.Structure, "kind"), null), e.kind);
    const regex = try send(&e, false, "{" ++ user ++ ",\"guided_regex\":\"(a)\\\\1\"}");
    try std.testing.expectEqual(@as(?u16, 400), regex.status);
    try std.testing.expect(std.mem.startsWith(u8, regex.text(), "guided_regex: "));
}

test "an engine without structures refuses them" {
    var e: Engine = .{ .structures = false };
    const seen = try send(&e, false, "{" ++ user ++ ",\"response_format\":{\"type\":\"json_object\"}}");
    try std.testing.expectEqual(@as(?u16, 400), seen.status);
    try std.testing.expect(std.mem.indexOf(u8, seen.text(), "not supported by this engine") != null);
}

const tools = ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}},{\"type\":\"function\",\"function\":{\"name\":\"ping\"}}]";

test "a required call under the DeepSeek-V4.1 dialect is a DSML calls grammar over the offered tools" {
    var e: Engine = .{};
    const seen = try send(&e, true, "{" ++ user ++ tools ++ ",\"tool_choice\":\"required\"}");
    try std.testing.expectEqual(@as(?u16, 200), seen.status);
    try std.testing.expectEqual(.calls, e.kind.?);
    try std.testing.expect(std.mem.indexOf(u8, e.text[0..e.len], "get_weather") != null and std.mem.indexOf(u8, e.text[0..e.len], "ping") != null);
    // a named function: that tool alone
    e.kind = null;
    const named = try send(&e, true, "{" ++ user ++ tools ++ ",\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"ping\"}}}");
    try std.testing.expectEqual(@as(?u16, 200), named.status);
    try std.testing.expectEqual(.calls, e.kind.?);
    try std.testing.expect(std.mem.indexOf(u8, e.text[0..e.len], "ping") != null and std.mem.indexOf(u8, e.text[0..e.len], "get_weather") == null);
    // parallel_tool_calls false: the grammar holds one call
    const one = try send(&e, true, "{" ++ user ++ tools ++ ",\"tool_choice\":\"required\",\"parallel_tool_calls\":false}");
    try std.testing.expectEqual(@as(?u16, 200), one.status);
    try std.testing.expect(std.mem.indexOf(u8, e.text[0..e.len], "\"parallel_tool_calls\": false") != null);
    // without structures the engine cannot hold the call: refused as before
    var off: Engine = .{ .structures = false };
    const refused = try send(&off, true, "{" ++ user ++ tools ++ ",\"tool_choice\":\"required\"}");
    try std.testing.expectEqual(@as(?u16, 400), refused.status);
}
