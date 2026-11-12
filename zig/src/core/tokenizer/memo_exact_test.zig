//! The memo's ids equal a fresh encode's on DeepSeek-shaped tokenizers: raw and normalized markers, a raw image
//! placeholder, a normalizer that composes or lengthens text, and invalid UTF-8 next to every kind of segment.
const std = @import("std");
const tokenizer = @import("tokenizer.zig");
const steps = @import("steps.zig");
const memo = @import("memo.zig");

// raw special, normalized chat markers (as DeepSeek-V4's), a raw non-special image placeholder (as
// <｜deepseek_image｜>), strip tokens, and a normalized one whose spelling NFC composes
const added =
    \\"added_tokens":[{"id":1000,"content":"<|bos|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
    \\ {"id":1001,"content":"<|User|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false},
    \\ {"id":1002,"content":"<|Assistant|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false},
    \\ {"id":1003,"content":"<think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false},
    \\ {"id":1004,"content":"</think>","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false},
    \\ {"id":1005,"content":"<|image|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":false},
    \\ {"id":1006,"content":"<|l|>","single_word":false,"lstrip":true,"rstrip":false,"normalized":false,"special":true},
    \\ {"id":1007,"content":"<|r|>","single_word":false,"lstrip":false,"rstrip":true,"normalized":false,"special":true},
    \\ {"id":1008,"content":"<|é|>","single_word":false,"lstrip":false,"rstrip":false,"normalized":true,"special":false}],
;

/// Byte-level BPE over every byte, NFC first, digits and words split as DeepSeek-V4's pre-tokenizer does.
fn byteLevel(a: std.mem.Allocator) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "{" ++ added ++
        \\ "normalizer":{"type":"NFC"},
        \\ "pre_tokenizer":{"type":"Sequence","pretokenizers":[{"type":"Split","pattern":{"Regex":"\\p{N}{1,3}"},"behavior":"Isolated","invert":false},
        \\  {"type":"Split","pattern":{"Regex":" ?\\S+|\\s+"},"behavior":"Isolated","invert":false},{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]},
        \\ "decoder":{"type":"ByteLevel"},
        \\ "model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":"","end_of_word_suffix":"","fuse_unk":false,"byte_fallback":false,"ignore_merges":false,"vocab":{
    );
    for (steps.byte_chars, 0..) |c, b| {
        var buf: [4]u8 = undefined;
        const n = try std.unicode.utf8Encode(c, &buf);
        const s = buf[0..n];
        try out.print(a, "\"{s}{s}\":{d},", .{ if (std.mem.eql(u8, s, "\"") or std.mem.eql(u8, s, "\\")) "\\" else "", s, b });
    }
    try out.appendSlice(a,
        \\"Ġa":256,"ab":257,"Ġab":258,"ca":259,"12":260,"Ã©":261},"merges":[["Ġ","a"],"a b","Ġa b","c a","1 2",["Ã","©"]]}}
    );
    return out.items;
}

/// Metaspace prepended to the first segment alone, byte fallback, and a normalizer that lengthens a tab.
fn metaspace(a: std.mem.Allocator) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    try out.appendSlice(a, "{" ++ added ++
        \\ "normalizer":{"type":"Replace","pattern":{"String":"\t"},"content":"    "},
        \\ "pre_tokenizer":{"type":"Metaspace","replacement":"▁","prepend_scheme":"first","split":true},
        \\ "model":{"type":"BPE","unk_token":"<unk>","fuse_unk":true,"byte_fallback":true,"vocab":{"<unk>":0,
    );
    for (0..256) |b| try out.print(a, "\"<0x{X:0>2}>\":{d},", .{ b, b + 1 });
    try out.appendSlice(a,
        \\"▁":257,"a":258,"b":259,"c":260,"ab":261,"▁a":262,"▁ab":263},"merges":["a b","▁ a","▁a b"]}}
    );
    return out.items;
}

const pieces = [_][]const u8{
    "<|bos|>",
    "<|User|>",
    "<|Assistant|>",
    "<think>",
    "</think>",
    "<|image|>",
    "<|l|>",
    "<|r|>",
    "<|e\u{301}|>",
    "<|\u{e9}|>",
    "e\u{301}t\u{e9} ",
    "\u{301}",
    "中文字符",
    "\u{1f642}",
    "12345 ",
    "  ",
    "\t",
    "\n",
    "\xff",
    "\xe4\xb8",
    "\xcc",
    " ",
    "x",
    "ab abc cab ",
};

fn text(a: std.mem.Allocator, r: std.Random) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    for (0..1 + r.uintLessThan(usize, 10)) |_| {
        if (r.uintLessThan(u8, 4) == 0) {
            // a body long enough for the memo, so the same one comes back in other texts and places
            for (0..10 + r.uintLessThan(usize, 2)) |_| try out.appendSlice(a, "ab abc cab ");
        } else try out.appendSlice(a, pieces[r.uintLessThan(usize, pieces.len)]);
    }
    return out.items;
}

test "the memo's ids equal a fresh encode's around markers, an image placeholder, normalization and broken UTF-8" {
    const gpa = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var prng = std.Random.DefaultPrng.init(0x7e57);
    const r = prng.random();
    var texts: [400][]const u8 = undefined;
    for (&texts) |*s| s.* = try text(a, r);
    for ([_][]const u8{ try byteLevel(a), try metaspace(a) }) |json| {
        var t = try tokenizer.parse(gpa, json);
        defer t.deinit();
        const image = t.specialTokenId("<|image|>").?;
        var fresh: [texts.len][]const u32 = undefined;
        for (texts, &fresh) |s, *ids| {
            ids.* = try t.encode(a, s);
            // each placeholder the text spells is one id, whatever stands around it
            try std.testing.expectEqual(std.mem.count(u8, s, "<|image|>"), std.mem.count(u32, ids.*, &.{image}));
        }
        // a memo that keeps everything, read in both orders, and one that evicts at every turn
        for ([_]usize{ 1 << 20, 3 * memo.bytes(memo.min_bytes, 32).? }) |cap| {
            t.memo = try tokenizer.Memo.create(gpa, cap);
            defer {
                t.memo.?.destroy();
                t.memo = null;
            }
            for (0..2) |round| for (0..texts.len) |k| {
                const i = if (round == 0) k else texts.len - 1 - k;
                const kept = try t.encode(gpa, texts[i]);
                defer gpa.free(kept);
                try std.testing.expectEqualSlices(u32, fresh[i], kept);
            };
            try std.testing.expect(t.memo.?.hits >= @as(u64, if (cap == 1 << 20) 400 else 10)); // segments came from the memo
        }
    }
}
