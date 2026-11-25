//! shared_messages: the end of a chat request's first k messages as a shared cut (messagesPrefixLen, chat.prepare).
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const errors = @import("errors.zig");
const model_text = @import("model_text.zig");
const prompt = @import("prompt.zig");
const routes = @import("routes.zig");
const Conn = @import("http_conn.zig").Conn;

/// One byte per character; a tools block first, then each message as a role letter, its content and a newline.
const RoleText = struct {
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
        for (ids, decoded) |id, *byte| byte.* = @truncate(id);
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

    fn render(_: *anyopaque, a: std.mem.Allocator, messages: json.Value, o: model_text.RenderOptions, _: *[]const u8) model_text.Error![]u8 {
        var out: std.ArrayList(u8) = .empty;
        if (o.tools) |tools| if (tools == .array) {
            try out.append(a, 'K');
            for (tools.array) |t| try out.print(a, "{s};", .{textOf(t.get("function") orelse t, "name")});
            try out.append(a, '\n');
        };
        if (messages == .array) for (messages.array) |m| {
            const role = textOf(m, "role");
            try out.append(a, if (role.len > 0) std.ascii.toUpper(role[0]) else '?');
            try out.print(a, "{s}\n", .{textOf(m, "content")});
        };
        if (o.add_generation_prompt) try out.append(a, 'A');
        return out.items;
    }

    fn templateSource(_: *anyopaque) []const u8 {
        return "";
    }
};

fn textOf(v: json.Value, key: []const u8) []const u8 {
    const x = v.get(key) orelse return "";
    return if (x == .string) x.string else "";
}

/// Keeps the shared cuts the server submits, then finishes at once.
const Capture = struct {
    shared: []u32 = &.{},

    fn engine(e: *@This()) api.Engine {
        return .{ .ctx = e, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(_: *anyopaque) api.Info {
        return .{};
    }

    fn submit(ctx: *anyopaque, id: api.Id, request: *const api.Request, sink: api.Sink) api.SubmitError!void {
        const e: *Capture = @ptrCast(@alignCast(ctx));
        std.testing.allocator.free(e.shared);
        e.shared = std.testing.allocator.dupe(u32, request.shared_prefixes) catch return error.Busy;
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

fn post(srv: *server.Server, body: []const u8) !u16 {
    var pair: [2]std.c.fd_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer _ = std.c.close(pair[0]);
    defer _ = std.c.close(pair[1]);
    const head = try std.fmt.allocPrint(std.testing.allocator, "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Length: {d}\r\n\r\n", .{body.len});
    defer std.testing.allocator.free(head);
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
    var tmp: [4096]u8 = undefined;
    var pollfd = [_]std.posix.pollfd{.{ .fd = pair[0], .events = std.posix.POLL.IN, .revents = 0 }};
    if (try std.posix.poll(&pollfd, 500) == 0) return error.NoReply;
    const n = try std.posix.read(pair[0], &tmp);
    if (n < "HTTP/1.1 200".len) return error.BadReply;
    return std.fmt.parseInt(u16, tmp["HTTP/1.1 ".len..][0..3], 10);
}

const system_text: *const [600]u8 = &@as([600]u8, @splat('s'));
const reading: *const [400]u8 = &@as([400]u8, @splat('r'));
const tools_json =
    \\[{"type":"function","function":{"name":"look","parameters":{"type":"object","properties":{}}}}]
;
/// A system message, a shared reading, a reply and the new question: k = 2 shares the first two.
const messages_json = "[{\"role\":\"system\",\"content\":\"" ++ system_text ++ "\"},{\"role\":\"user\",\"content\":\"" ++
    reading ++ "\"},{\"role\":\"assistant\",\"content\":\"ok\"},{\"role\":\"user\",\"content\":\"go\"}]";
const short_json =
    \\[{"role":"user","content":"hi"},{"role":"assistant","content":"ok"},{"role":"user","content":"go"}]
;
/// Rendered lengths: system message and reading (role letter, text, newline), tools block ("K", "look;", newline).
const system_len = 1 + system_text.len + 1;
const reading_len = 1 + reading.len + 1;
const tools_len = 1 + "look;".len + 1;
/// The probe is a user turn, so a cut before a user message also takes that message's role letter.
const user_mark = 1;

fn testServer(text: *RoleText, backend: *Capture) !*server.Server {
    return server.Server.init(std.testing.allocator, std.testing.io, backend.engine(), text.text(), .{
        .served_name = "m",
        .model_ids = &.{"m"},
        .enable_thinking = false,
        .use_drafts = false,
    }, null);
}

fn parsed(a: std.mem.Allocator, text: []const u8) !json.Value {
    return switch (try json.parse(a, text)) {
        .ok => |v| v,
        .err => error.BadJson,
    };
}

/// messagesPrefixLen for `k` on `conversation`, with or without the tools.
fn prefixLen(srv: *server.Server, conversation: []const u8, k: usize, with_tools: bool) !usize {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: errors.Cx = .{ .a = arena.allocator() };
    const messages = try parsed(cx.a, conversation);
    const tools: []const json.Value = if (with_tools) (try parsed(cx.a, tools_json)).array else &.{};
    const ids = try prompt.renderIds(srv, &cx, messages, tools, false, null, true, null);
    return prompt.messagesPrefixLen(srv, &cx, messages, k, tools, ids, false, null, null);
}

test "messagesPrefixLen: k = 0, k at or past the end and a cut under 512 add nothing" {
    var text: RoleText = .{};
    var backend: Capture = .{};
    defer std.testing.allocator.free(backend.shared);
    const srv = try testServer(&text, &backend);
    defer srv.deinit();
    try std.testing.expectEqual(@as(usize, 0), try prefixLen(srv, messages_json, 0, false));
    try std.testing.expectEqual(@as(usize, 0), try prefixLen(srv, messages_json, 4, false));
    try std.testing.expectEqual(@as(usize, 0), try prefixLen(srv, messages_json, 9, false));
    try std.testing.expectEqual(@as(usize, 0), try prefixLen(srv, short_json, 1, false));
    try std.testing.expectEqual(@as(usize, 0), try prefixLen(srv, short_json, 2, true));
}

test "messagesPrefixLen: the cut ends at the k-th message, past the tools block when tools are offered" {
    var text: RoleText = .{};
    var backend: Capture = .{};
    defer std.testing.allocator.free(backend.shared);
    const srv = try testServer(&text, &backend);
    defer srv.deinit();
    try std.testing.expectEqual(@as(usize, system_len + user_mark), try prefixLen(srv, messages_json, 1, false));
    try std.testing.expectEqual(@as(usize, system_len + reading_len), try prefixLen(srv, messages_json, 2, false));
    try std.testing.expectEqual(@as(usize, tools_len + system_len + reading_len), try prefixLen(srv, messages_json, 2, true));
    const reply_len = 1 + "ok".len + 1;
    try std.testing.expectEqual(@as(usize, tools_len + system_len + reading_len + reply_len + user_mark), try prefixLen(srv, messages_json, 3, true));
}

test "shared_messages reaches the request's shared_prefixes; a non-integer, zero, negative or too large k is ignored" {
    var text: RoleText = .{};
    var backend: Capture = .{};
    defer std.testing.allocator.free(backend.shared);
    const srv = try testServer(&text, &backend);
    defer srv.deinit();
    const cut: u32 = tools_len + system_len + reading_len;
    const sys: u32 = tools_len + system_len + user_mark; // the server's own cut, at the end of the system block
    const Case = struct { k: []const u8, want: []const u32 };
    const cases = [_]Case{
        .{ .k = "", .want = &.{sys} },
        .{ .k = ",\"shared_messages\":2", .want = &.{ sys, cut } },
        .{ .k = ",\"shared_messages\":2.0", .want = &.{sys} },
        .{ .k = ",\"shared_messages\":\"2\"", .want = &.{sys} },
        .{ .k = ",\"shared_messages\":true", .want = &.{sys} },
        .{ .k = ",\"shared_messages\":0", .want = &.{sys} },
        .{ .k = ",\"shared_messages\":-1", .want = &.{sys} },
        .{ .k = ",\"shared_messages\":4", .want = &.{sys} },
    };
    for (cases) |case| {
        const body = try std.fmt.allocPrint(std.testing.allocator, "{{\"model\":\"m\",\"max_tokens\":2,\"tools\":{s},\"messages\":{s}{s}}}", .{ tools_json, messages_json, case.k });
        defer std.testing.allocator.free(body);
        try std.testing.expectEqual(@as(u16, 200), try post(srv, body));
        try std.testing.expectEqualSlices(u32, case.want, backend.shared);
    }
}
