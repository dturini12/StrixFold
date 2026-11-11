//! A sampled request's top_k against an engine whose sampler draws only among its best top_k tokens (Info.top_k_max,
//! Info.needs_top_k; DeepSeek-V4.1's: 1 to 56, no top_k-off path): refused with a 400 that names top_k, before any
//! header and before the engine sees it (U49: it was a 500 from the engine, named as the TP-world setting).
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const openai = @import("openai.zig");
const Conn = @import("http_conn.zig").Conn;

/// Bytes as tokens; every prompt renders as "prompt".
const TestText = struct {
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

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// An engine with the given sampler limits that keeps the last request's sampling; it answers "ok".
const Engine = struct {
    top_k_max: u32 = 56,
    needs_top_k: bool = true,
    submitted: usize = 0,
    sampling: ?api.Sampling = null,

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 64, .top_k_max = e.top_k_max, .needs_top_k = e.needs_top_k };
    }

    fn submit(ctx: *anyopaque, id: api.Id, r: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Engine = @ptrCast(@alignCast(ctx));
        e.submitted += 1;
        e.sampling = r.sampling;
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'o', 'k' } });
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

/// A reply's status and error message, and whether a stream's head went out.
const Seen = struct {
    status: ?u16 = null,
    opened: bool = false,
    message: [512]u8 = undefined,
    len: usize = 0,

    fn out(c: *Seen) openai.Out {
        return .{ .ctx = c, .vt = &.{ .open = open, .event = event, .reply = reply } };
    }

    fn open(ctx: *anyopaque) error{Closed}!void {
        const c: *Seen = @ptrCast(@alignCast(ctx));
        c.opened = true;
    }

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

    fn says(c: *const Seen, part: []const u8) bool {
        return std.mem.indexOf(u8, c.text(), part) != null;
    }
};

fn send(e: *Engine, body: []const u8) !Seen {
    return sendWith(e, body, null);
}

/// `send` on a server whose sampling defaults are `defaults` (the serve flags', a JSON object).
fn sendWith(e: *Engine, body: []const u8, defaults: ?[]const u8) !Seen {
    var text: TestText = .{};
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, e.engine(), text.text(), .{
        .served_name = "test-model",
        .model_ids = &.{"test-model"},
        .enable_thinking = false,
        .use_drafts = false,
        .default_sampling = if (defaults) |d| (try json.parse(a, d)).ok else null,
    }, null);
    defer srv.deinit();
    var conn: Conn = .{ .fd = -1, .peer = "", .buf = &.{}, .gpa = a };
    var seen: Seen = .{};
    openai.run(srv, a, seen.out(), .{ .conn = &conn }, true, (try json.parse(a, body)).ok);
    return seen;
}

const user = "{\"model\":\"test-model\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":8";

test "a sampled request without top_k is a 400 that names top_k, not a failure in the engine" {
    var e: Engine = .{};
    inline for ([_][]const u8{ ",\"temperature\":0.7}", ",\"temperature\":1.5,\"top_p\":1.0}", ",\"temperature\":0.7,\"top_k\":0}", ",\"temperature\":0.7,\"top_k\":-1}", ",\"temperature\":0.7,\"stream\":true}" }) |tail| {
        const seen = try send(&e, user ++ tail);
        try std.testing.expectEqual(@as(?u16, 400), seen.status);
        try std.testing.expect(!seen.opened);
        try std.testing.expect(std.mem.startsWith(u8, seen.text(), "top_k: "));
        try std.testing.expect(seen.says("from 1 to 56"));
        try std.testing.expect(!seen.says("TF_TP_WORLD"));
    }
    try std.testing.expectEqual(@as(usize, 0), e.submitted);
}

test "top_k above the engine's sampler limit is a 400 that names it" {
    var e: Engine = .{};
    const seen = try send(&e, user ++ ",\"temperature\":0.7,\"top_k\":57}");
    try std.testing.expectEqual(@as(?u16, 400), seen.status);
    try std.testing.expect(seen.says("top_k 57") and seen.says("from 1 to 56"));
    try std.testing.expectEqual(@as(usize, 0), e.submitted);
}

test "top_k within the limit, and greedy without one, reach the engine" {
    var e: Engine = .{};
    const sampled = try send(&e, user ++ ",\"temperature\":0.7,\"top_k\":56,\"seed\":3}");
    try std.testing.expectEqual(@as(?u16, 200), sampled.status);
    try std.testing.expectEqual(@as(u32, 56), e.sampling.?.top_k);
    const greedy = try send(&e, user ++ ",\"temperature\":0}");
    try std.testing.expectEqual(@as(?u16, 200), greedy.status);
    try std.testing.expectEqual(@as(?api.Sampling, null), e.sampling);
    try std.testing.expectEqual(@as(usize, 2), e.submitted);
}

test "an engine that samples with top_k off serves a request without one" {
    var e: Engine = .{ .top_k_max = 0, .needs_top_k = false };
    const seen = try send(&e, user ++ ",\"temperature\":0.7,\"top_p\":0.9}");
    try std.testing.expectEqual(@as(?u16, 200), seen.status);
    try std.testing.expectEqual(@as(u32, 0), e.sampling.?.top_k);
}

test "draft_temperature: the request's own, else the server's; never for a greedy request; refused outside 0 to 100" {
    var e: Engine = .{};
    const sampled = ",\"temperature\":0.7,\"top_k\":20,\"seed\":3";
    // without it anywhere: 0 (the drafts are the drafter's best tokens)
    try std.testing.expectEqual(@as(?u16, 200), (try send(&e, user ++ sampled ++ "}")).status);
    try std.testing.expectEqual(@as(f64, 0), e.sampling.?.draft_temperature);
    try std.testing.expectEqual(@as(?u16, 200), (try send(&e, user ++ sampled ++ ",\"draft_temperature\":0.75}")).status);
    try std.testing.expectEqual(@as(f64, 0.75), e.sampling.?.draft_temperature);
    // the server's default (--draft-temperature) for a request without one, beside its default temperature
    const defaults = "{\"temperature\":1.0,\"top_k\":20,\"draft_temperature\":0.5}";
    try std.testing.expectEqual(@as(?u16, 200), (try sendWith(&e, user ++ "}", defaults)).status);
    try std.testing.expectEqual(@as(f64, 0.5), e.sampling.?.draft_temperature);
    try std.testing.expectEqual(@as(f64, 1.0), e.sampling.?.temperature);
    try std.testing.expectEqual(@as(?u16, 200), (try sendWith(&e, user ++ ",\"draft_temperature\":1}", defaults)).status);
    try std.testing.expectEqual(@as(f64, 1.0), e.sampling.?.draft_temperature);
    try std.testing.expectEqual(@as(?u16, 200), (try sendWith(&e, user ++ ",\"draft_temperature\":0}", defaults)).status);
    try std.testing.expectEqual(@as(f64, 0), e.sampling.?.draft_temperature);
    // a greedy request stays greedy: no sampling, so nothing to draw drafts with
    try std.testing.expectEqual(@as(?u16, 200), (try sendWith(&e, user ++ ",\"temperature\":0}", defaults)).status);
    try std.testing.expectEqual(@as(?api.Sampling, null), e.sampling);
    const before = e.submitted;
    inline for ([_][]const u8{ ",\"draft_temperature\":-0.5}", ",\"draft_temperature\":101}", ",\"draft_temperature\":\"warm\"}" }) |tail| {
        const seen = try send(&e, user ++ sampled ++ tail);
        try std.testing.expectEqual(@as(?u16, 400), seen.status);
        try std.testing.expect(seen.says("draft_temperature"));
    }
    try std.testing.expectEqual(before, e.submitted);
}
