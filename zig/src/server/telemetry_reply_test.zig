//! An engine's per-request telemetry (Stats.telemetry_json) reaches the reply as speculative.proposer intact: the
//! reply is written after the request's mailbox, which held the engine's bytes, has been freed.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const model_text = @import("model_text.zig");
const routes = @import("routes.zig");
const Conn = @import("http_conn.zig").Conn;

/// Byte tokens and a one-line prompt.
const ByteText = struct {
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
        return a.dupe(u8, "hi\n");
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

/// Finishes each request at once with telemetry in its own buffer, freed as soon as the event returns (as the lane
/// host frees its job).
const Telemetry = struct {
    const sent = "{\"rounds_by_rows\": {\"1\": [2, 30.500, 465.125]}, \"drafted\": 3, \"graphs\": {\"captured\": 1, \"replayed\": 4, \"eager\": 0}}";

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{};
    }

    fn submit(_: *anyopaque, id: api.Id, _: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const own = std.testing.allocator.dupe(u8, sent) catch return error.Busy;
        defer std.testing.allocator.free(own);
        sink.event(sink.ctx, id, &.{ .prefilled = 0 });
        sink.event(sink.ctx, id, &.{ .tokens = &.{ 'o', 'k' } });
        sink.event(sink.ctx, id, &.{ .finished = .{ .reason = .stop, .stats = .{ .rounds = 2, .drafted = 3, .telemetry_json = own } } });
    }

    fn cancel(_: *anyopaque, _: api.Id) void {}

    fn status(_: *anyopaque, out: *api.Status, _: []u32) void {
        out.* = .{};
    }

    fn memory(_: *anyopaque, _: bool) ?api.Memory {
        return null;
    }
};

/// The whole HTTP reply to one non-streamed request.
fn post(a: std.mem.Allocator, srv: *server.Server, body: []const u8) ![]u8 {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const head = try std.fmt.allocPrint(a, "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n", .{body.len});
    for ([_][]const u8{ head, body }) |part| {
        var sent: usize = 0;
        while (sent < part.len) {
            const n = std.c.write(pair[0], part[sent..].ptr, part.len - sent);
            if (n <= 0) return error.Closed;
            sent += @intCast(n);
        }
    }
    var conn = try Conn.init(std.testing.allocator, pair[1], "test");
    defer conn.deinit();
    conn.timeouts = .{ .idle_ms = 2000, .read_ms = 2000, .write_ms = 2000 };
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    if (try conn.readRequest(arena.allocator(), false) != .ready) return error.BadRequest;
    routes.dispatch(srv, &conn, arena.allocator());
    var got: std.ArrayList(u8) = .empty;
    var tmp: [4096]u8 = undefined;
    var pollfd = [_]std.posix.pollfd{.{ .fd = pair[0], .events = std.posix.POLL.IN, .revents = 0 }};
    while (try std.posix.poll(&pollfd, 200) > 0) {
        const n = try std.posix.read(pair[0], &tmp);
        if (n == 0) break;
        try got.appendSlice(a, tmp[0..n]);
    }
    return got.items;
}

test "an engine's telemetry reaches the reply's speculative.proposer intact after the mailbox is freed" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: ByteText = .{};
    var backend: Telemetry = .{};
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false }, null);
    defer srv.deinit();
    const reply = try post(a, srv, "{\"model\":\"m\",\"max_tokens\":2,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    try std.testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200"));
    const at = std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return error.BadReply;
    const parsed = try json.parse(a, reply[at + 4 ..]);
    try std.testing.expect(parsed == .ok);
    const proposer = parsed.ok.get("speculative").?.get("proposer") orelse return error.NoProposer;
    try std.testing.expectEqualStrings("3", proposer.get("drafted").?.int);
    try std.testing.expectEqualStrings("4", proposer.get("graphs").?.get("replayed").?.int);
    const rows = proposer.get("rounds_by_rows").?.get("1").?.array;
    try std.testing.expectEqual(@as(usize, 3), rows.len);
    try std.testing.expectEqualStrings("2", rows[0].int);
}

/// Reports more restored prompt tokens than the prompt holds (issue #1899: the TP=2 doubling).
const OverCached = struct {
    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{};
    }

    fn submit(_: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        sink.event(sink.ctx, id, &.{ .prefilled = @intCast(request.prompt.len + 5) });
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

test "cached tokens above the prompt are clamped to the prompt in the reply" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var text: ByteText = .{};
    var backend: OverCached = .{};
    var srv = try server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false }, null);
    defer srv.deinit();
    const reply = try post(a, srv, "{\"model\":\"m\",\"max_tokens\":2,\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}");
    try std.testing.expect(std.mem.startsWith(u8, reply, "HTTP/1.1 200"));
    const at = std.mem.indexOf(u8, reply, "\r\n\r\n") orelse return error.BadReply;
    const parsed = try json.parse(a, reply[at + 4 ..]);
    try std.testing.expect(parsed == .ok);
    const usage = parsed.ok.get("usage").?;
    const prompt = usage.get("prompt_tokens").?.int;
    try std.testing.expect(!std.mem.eql(u8, prompt, "0"));
    try std.testing.expectEqualStrings(prompt, usage.get("prompt_tokens_details").?.get("cached_tokens").?.int);
}
