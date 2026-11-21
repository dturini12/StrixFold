//! The preparation memo (--vision-image-memo) through prompt.prepare, with a fake vision family counting its prepares.
const std = @import("std");
const api = @import("engine_api");
const server = @import("root.zig");
const json = server.json;
const prompt = @import("prompt.zig");
const chat = @import("chat.zig");
const errors = @import("errors.zig");
const Rig = @import("text_rig_test.zig").Rig;

/// A chat template that only joins the messages (the placeholders are in their text).
const chat_template = "{% for message in messages %}<{{ message['role'] }}>{{ message['content'] }}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}";

/// Counts its prepares: a source starting "bad" is refused (problem in the prepare's allocator), else "payload:" ++ it.
const Fake = struct {
    source_prepare: bool = true,
    max_tokens: u32 = 0,
    max_pixels: u64 = 0,
    tokens: u32 = 3,
    pixels: u64 = 100,
    prepares: usize = 0,

    fn engine(f: *Fake) api.Engine {
        return .{ .ctx = f, .vtable = &.{ .info = info, .submit = submit, .cancel = cancel, .status = status, .memory = memory } };
    }

    fn info(ctx: *anyopaque) api.Info {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        return .{ .context_window = 1 << 16, .vision = .{ .ctx = f, .placeholder = Rig.image_placeholder, .token = Rig.image_id, .prepare = prepare, .max_tokens = f.max_tokens, .max_pixels = f.max_pixels, .source_prepare = f.source_prepare } };
    }

    fn prepare(ctx: *anyopaque, a: std.mem.Allocator, bytes: []const u8, problem: *[]const u8) api.Vision.PrepareError!api.Vision.Prepared {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.prepares += 1;
        if (std.mem.startsWith(u8, bytes, "bad")) {
            problem.* = try std.fmt.allocPrint(a, "image input: cannot read {s}", .{bytes});
            return error.BadImage;
        }
        const payload = try a.alignedAlloc(u8, .@"16", "payload:".len + bytes.len);
        @memcpy(payload[0.."payload:".len], "payload:");
        @memcpy(payload["payload:".len..], bytes);
        return .{ .tokens = f.tokens, .payload = payload, .pixels = f.pixels };
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

/// A server with `memo` (the --vision-image-memo count) over the deployed template and `fake`'s vision.
const Rigged = struct {
    rig: *Rig, // the server's text points into it
    srv: *server.Server,

    fn start(a: std.mem.Allocator, fake: *Fake, memo: u32) !Rigged {
        const rig = try a.create(Rig);
        rig.* = try Rig.init(a, chat_template);
        return .{ .rig = rig, .srv = try server.Server.init(std.testing.allocator, std.testing.io, fake.engine(), rig.text(), .{ .served_name = "m", .model_ids = &.{"m"}, .use_drafts = false, .enable_thinking = false, .image_memo = memo }, null) };
    }

    fn stop(r: *Rigged) void {
        r.srv.deinit();
    }
};

/// One user message with a placeholder for each of `images`, through prompt.prepare (the chat route's).
fn ask(r: *Rigged, a: std.mem.Allocator, cx: *errors.Cx, images: []const []const u8) errors.Refused!prompt.Rendered {
    var text: std.ArrayList(u8) = .empty;
    try text.appendSlice(a, "look:");
    for (images) |_| try text.print(a, " {s}", .{Rig.image_placeholder});
    const message = try json.newObject(a);
    try message.put(a, "role", .{ .string = "user" });
    try message.put(a, "content", .{ .string = text.items });
    const list = try a.alloc(json.Value, 1);
    list[0] = .{ .object = message };
    const input: chat.Input = .{ .messages = .{ .array = list }, .fields = .{ .object = try json.newObject(a) }, .images = images };
    return prompt.prepare(r.srv, cx, input, false, null);
}

fn expectToday(fake_payload_of: []const u8, img: api.Image, a: std.mem.Allocator) !void {
    const payload = try std.mem.concat(a, u8, &.{ "payload:", fake_payload_of });
    try std.testing.expectEqualStrings(payload, img.bytes);
    try std.testing.expectEqual(@as(usize, 0), img.source.len);
    try std.testing.expectEqualSlices(u8, &api.prompt_cache.imageDigest(payload), &img.digest.?);
}

test "memo off, or an engine that does not prepare from source: every span carries its payload and no source, as today" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pictures = [_][]const u8{ "one", "two", "one" };
    var off_ids: []const u32 = &.{};
    for ([_]struct { memo: u32, source_prepare: bool }{ .{ .memo = 0, .source_prepare = true }, .{ .memo = 0, .source_prepare = false }, .{ .memo = 8, .source_prepare = false } }) |case| {
        var fake: Fake = .{ .source_prepare = case.source_prepare };
        var r = try Rigged.start(a, &fake, case.memo);
        defer r.stop();
        try std.testing.expect(r.srv.memo == null);
        for (0..2) |turn| {
            var cx: errors.Cx = .{ .a = a };
            const got = try ask(&r, a, &cx, &pictures);
            try std.testing.expectEqual(@as(usize, 3), got.images.len);
            for (got.images, pictures) |img, p| try expectToday(p, img, a);
            // every picture prepared on every request
            try std.testing.expectEqual(3 * (turn + 1), fake.prepares);
            if (off_ids.len == 0) off_ids = got.ids else try std.testing.expectEqualSlices(u32, off_ids, got.ids);
        }
    }
}

test "memo on: a repeated picture is prepared once, its span carries the source, and the prompt is the one without memo" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const pictures = [_][]const u8{ "one", "two", "one" };
    var plain: Fake = .{};
    var off = try Rigged.start(a, &plain, 0);
    defer off.stop();
    var cx0: errors.Cx = .{ .a = a };
    const want = try ask(&off, a, &cx0, &pictures);

    var fake: Fake = .{};
    var r = try Rigged.start(a, &fake, 8);
    defer r.stop();
    try std.testing.expect(r.srv.memo != null);
    for (0..3) |_| {
        var cx: errors.Cx = .{ .a = a };
        const got = try ask(&r, a, &cx, &pictures);
        // "one" twice in the request and on every turn: prepared once; "two" once
        try std.testing.expectEqual(@as(usize, 2), fake.prepares);
        try std.testing.expectEqualSlices(u32, want.ids, got.ids);
        try std.testing.expectEqual(want.history_len, got.history_len);
        for (got.images, want.images, pictures) |img, w, p| {
            try std.testing.expectEqual(@as(usize, 0), img.bytes.len);
            try std.testing.expectEqualStrings(p, img.source);
            try std.testing.expectEqual(w.at, img.at);
            try std.testing.expectEqual(w.tokens, img.tokens);
            // the digest of the payload the server would have sent: the prompt caches' keys are unchanged
            try std.testing.expectEqualSlices(u8, &w.digest.?, &img.digest.?);
        }
    }
    try std.testing.expectEqual(@as(usize, 2), r.srv.memo.?.count(std.testing.io));
}

test "memo on: the limits take the memo's tokens and pixels, and refuse at the picture that crosses them" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fake: Fake = .{ .tokens = 3, .pixels = 100, .max_tokens = 7, .max_pixels = 250 };
    var r = try Rigged.start(a, &fake, 8);
    defer r.stop();
    var cx: errors.Cx = .{ .a = a };
    _ = try ask(&r, a, &cx, &.{ "one", "two" }); // 6 tokens, 200 pixels: under both
    try std.testing.expectEqual(@as(usize, 2), fake.prepares);
    // a fresh preparation would now give 1 token and 1 pixel; the remembered pictures keep their 3 and 100
    fake.tokens = 1;
    fake.pixels = 1;
    const got = try ask(&r, a, &cx, &.{ "two", "one" });
    try std.testing.expectEqual(@as(u32, 3), got.images[0].tokens);
    try std.testing.expectEqual(@as(usize, 2), fake.prepares);
    // three remembered pictures: 300 pixels pass 250 at the third (the pixel limit is checked first)
    try std.testing.expectError(error.Refused, ask(&r, a, &cx, &.{ "one", "two", "one" }));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "300 pixels by image 3 of 3") != null);
    try std.testing.expectEqual(@as(usize, 2), fake.prepares);
    // tokens: 3 + 3 + 3 = 9 pass 7 at the third when pixels allow it
    var roomy: Fake = .{ .tokens = 3, .pixels = 100, .max_tokens = 7 };
    var r2 = try Rigged.start(a, &roomy, 8);
    defer r2.stop();
    _ = try ask(&r2, a, &cx, &.{"one"});
    roomy.tokens = 1;
    try std.testing.expectError(error.Refused, ask(&r2, a, &cx, &.{ "one", "one", "one" }));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "9 image tokens by image 3 of 3") != null);
    try std.testing.expectEqual(@as(usize, 1), roomy.prepares);
}

test "memo on: a refused picture is not remembered, and its refusal outlives the scratch it was prepared in" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fake: Fake = .{};
    var r = try Rigged.start(a, &fake, 8);
    defer r.stop();
    for (1..3) |n| {
        var cx: errors.Cx = .{ .a = a };
        try std.testing.expectError(error.Refused, ask(&r, a, &cx, &.{ "one", "bad-png" }));
        try std.testing.expectEqualStrings("image input: cannot read bad-png", cx.message);
        try std.testing.expect(cx.kind == .request);
        // "one" prepared once; "bad-png" on every request
        try std.testing.expectEqual(1 + n, fake.prepares);
    }
    try std.testing.expectEqual(@as(usize, 1), r.srv.memo.?.count(std.testing.io));
}

test "memo on: a full memo is cleared before the next new picture, and what it held is prepared again" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var fake: Fake = .{};
    var r = try Rigged.start(a, &fake, 2);
    defer r.stop();
    var cx: errors.Cx = .{ .a = a };
    _ = try ask(&r, a, &cx, &.{ "one", "two" });
    _ = try ask(&r, a, &cx, &.{ "one", "two" });
    try std.testing.expectEqual(@as(usize, 2), fake.prepares);
    try std.testing.expectEqual(@as(u64, 0), r.srv.memo.?.clears);
    _ = try ask(&r, a, &cx, &.{"three"}); // full: cleared, then "three" alone
    try std.testing.expectEqual(@as(u64, 1), r.srv.memo.?.clears);
    try std.testing.expectEqual(@as(usize, 1), r.srv.memo.?.count(std.testing.io));
    const got = try ask(&r, a, &cx, &.{"one"});
    try std.testing.expectEqual(@as(usize, 4), fake.prepares);
    try std.testing.expectEqualStrings("one", got.images[0].source);
}
