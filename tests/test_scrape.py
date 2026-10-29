import io

from wthwid.scrape import extract_pairs, iter_revisions, site_url
from wthwid.summaries import clean_summary

DUMP = """<mediawiki xmlns="http://www.mediawiki.org/xml/export-0.11/">
  <siteinfo><sitename>Wikipedia</sitename></siteinfo>
  <page>
    <title>Cat</title><ns>0</ns><id>7</id>
    <revision>
      <id>1</id><timestamp>2010-01-01T00:00:00Z</timestamp>
      <contributor><username>A</username><id>99</id></contributor>
      <comment>Created page with "x"</comment><model>wikitext</model>
      <text>A cat is an animal.</text>
    </revision>
    <revision>
      <id>2</id><parentid>1</parentid><timestamp>2011-01-01T00:00:00Z</timestamp>
      <contributor><ip>1.2.3.4</ip></contributor>
      <minor/>
      <comment>/* Intro */ fix typo</comment><model>wikitext</model>
      <text>A cat is a animal.</text>
    </revision>
    <revision>
      <id>3</id><parentid>2</parentid><timestamp>2012-01-01T00:00:00Z</timestamp>
      <contributor><username>B</username><id>5</id></contributor>
      <comment deleted="deleted" /><model>wikitext</model>
      <text>A cat is an animal.</text>
    </revision>
    <revision>
      <id>4</id><parentid>3</parentid><timestamp>2013-01-01T00:00:00Z</timestamp>
      <contributor><username>C</username><id>6</id></contributor>
      <comment>add fact</comment><model>wikitext</model>
      <text>A cat is an animal.
Cats purr.</text>
    </revision>
  </page>
  <page>
    <title>Talk:Cat</title><ns>1</ns><id>8</id>
    <revision>
      <id>5</id><parentid>4</parentid><timestamp>2013-01-01T00:00:00Z</timestamp>
      <contributor><username>C</username><id>6</id></contributor>
      <comment>talk</comment><model>wikitext</model><text>hi</text>
    </revision>
  </page>
</mediawiki>"""


def pairs(**kw):
    args = dict(wiki="simplewiki", namespaces={0}, since=None,
                max_diff_chars=10_000, keep_bots=False)
    args.update(kw)
    return list(extract_pairs(iter_revisions(io.BytesIO(DUMP.encode())), **args))


def test_iter_revisions_fields():
    revs = list(iter_revisions(io.BytesIO(DUMP.encode())))
    assert [r["id"] for r in revs] == [1, 2, 3, 4, 5]
    assert revs[1]["parentid"] == 1 and revs[1]["minor"] is True
    assert revs[1]["page_id"] == 7  # contributor <id> must not leak in
    assert revs[2]["comment"] is None
    assert revs[4]["title"] == "Talk:Cat" and revs[4]["ns"] == 1


def test_extract_pairs():
    got = pairs()
    # rev 1: creation, rev 3: deleted comment, rev 5: talk namespace.
    assert [p["rev_id"] for p in got] == [2, 4]
    assert got[0]["summary"] == "fix typo"
    assert got[0]["diff"] == "@@ -1 +1 @@\n-A cat is an animal.\n+A cat is a animal."
    assert got[0]["source"] == "https://simple.wikipedia.org/w/index.php?diff=2&oldid=1"
    assert got[1]["summary"] == "add fact"


def test_since_filter():
    assert [p["rev_id"] for p in pairs(since="2012")] == [4]


def test_clean_summary():
    assert clean_summary("/* History */ added dates") == "added dates"
    assert clean_summary("/* History */") is None
    assert clean_summary("Undid revision 123 by [[Special:Contributions/X|X]]") is None
    assert clean_summary("Reverted edits by X") is None
    assert clean_summary("←Created page with 'foo'") is None
    assert clean_summary("r2.6.4) (Robot: Adding [[su:Maroko]]") is None
    assert clean_summary("Bot: fixing links", keep_bots=True) == "Bot: fixing links"
    assert clean_summary("rv") is None
    assert clean_summary("rvb fix") == "rvb fix"
    assert clean_summary("thanks [[User:Foo]]") is None
    assert clean_summary("  copyedit  ") == "copyedit"
    assert clean_summary(None) is None


def test_site_url():
    assert site_url("enwiki") == "https://en.wikipedia.org"
    assert site_url("zh_min_nanwiki") == "https://zh-min-nan.wikipedia.org"
