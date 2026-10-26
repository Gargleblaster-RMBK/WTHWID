from wthwid.diffs import compact_diff, make_diff


def test_make_diff_is_plain_unified_without_headers():
    d = make_diff("a\nb\nc\n", "a\nB\nc\n")
    assert d.splitlines() == ["@@ -2 +2 @@", "-b", "+B"]


def test_identical_text_gives_empty_diff():
    assert make_diff("same\n", "same\n") == ""


def test_word_diff_for_changed_line():
    d = make_diff("The cat sat on the mat.", "The dog sat on the mat.")
    assert compact_diff(d) == "~ The [-cat-] {+dog+} sat on the mat."


def test_pure_additions_and_removals():
    d = make_diff("one\ntwo\n", "one\nthree\nfour\n")
    out = compact_diff(d).splitlines()
    assert out == ["~ [-two-] {+three+}", "+ four"]
    assert compact_diff(make_diff("x\ny\n", "x\n")) == "- y"


def test_removed_line_starting_with_dashes_is_not_a_header():
    # "-- sig" removed renders as "--- sig" in the diff body.
    d = make_diff("keep\n-- sig\n", "keep\n")
    assert "--- sig" in d
    assert compact_diff(d) == "- -- sig"


def test_accepts_git_diff_with_headers_and_context():
    git = "\n".join([
        "diff --git a/README.md b/README.md",
        "index 1111111..2222222 100644",
        "--- a/README.md",
        "+++ b/README.md",
        "@@ -1,3 +1,3 @@",
        " # Title",
        "-Hello wrld",
        "+Hello world",
        " end",
        "\\ No newline at end of file",
    ])
    assert compact_diff(git) == "~ Hello [-wrld-] {+world+}"


def test_long_context_is_trimmed():
    words = " ".join(f"w{i}" for i in range(40))
    d = make_diff(words + " old", words + " new")
    out = compact_diff(d)
    assert out.startswith("~ … ")
    assert out.endswith("[-old-] {+new+}")
    assert "w0 " not in out


def test_max_chars_truncates():
    d = make_diff("", "\n".join("line %d" % i for i in range(1000)))
    assert len(compact_diff(d, max_chars=100)) == 100
