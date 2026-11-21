//! A test text model for the server's prompt tests.
const std = @import("std");
const template = @import("template");
const model_text = @import("model_text.zig");
const HfText = @import("hf_text.zig").HfText;

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
