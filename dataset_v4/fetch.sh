#!/usr/bin/env bash
#
# fetch.sh — download the curated v4 dataset.
#
# 12 files in 4 categories of 3:
#   objects/      — single-object Unsplash photographs (proxy for
#                   sprites/icons; Wikimedia URLs failed during URL
#                   hunt and free pixel-art at predictable URLs is
#                   scarce, so we use simple-composition real photos
#                   as the "small visual surface" stand-in).
#   ui/           — Unsplash photos of monitors / code on screen
#                   (proxy for product UI screenshots; same caveat —
#                   real product screenshots aren't easily available
#                   at stable public URLs).
#   architecture/ — Unsplash architecture photographs.
#   pdfs/         — long, hard-to-understand technical PDFs from arXiv.
#

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

mkdir -p objects ui architecture pdfs

ua='Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15'

fetch() {
    local url="$1" dst="$2"
    if [[ -s "$dst" ]]; then
        echo "  skip (have): $dst"
        return 0
    fi
    echo "  GET $url"
    curl -fsSL -A "$ua" -o "$dst.tmp" "$url" || { rm -f "$dst.tmp"; return 1; }
    mv "$dst.tmp" "$dst"
}

unsplash() {
    fetch "https://images.unsplash.com/photo-$1?w=1280&q=80" "$2"
}

echo "[objects]"
unsplash 1495474472287-4d71bcdd2085 objects/01.jpg
unsplash 1559056199-641a0ac8b55e    objects/02.jpg
unsplash 1495774856032-8b90bbb32b32 objects/03.jpg

echo "[ui]"
unsplash 1517245386807-bb43f82c33c4 ui/01.jpg
unsplash 1486312338219-ce68d2c6f44d ui/02.jpg
unsplash 1581291518633-83b4ebd1d83e ui/03.jpg

echo "[architecture]"
unsplash 1487958449943-2429e8be8625 architecture/01.jpg
unsplash 1496564203457-11bb12075d90 architecture/02.jpg
unsplash 1448630360428-65456885c650 architecture/03.jpg

echo "[pdfs]"
fetch "https://arxiv.org/pdf/1706.03762.pdf" pdfs/attention_is_all_you_need.pdf
fetch "https://arxiv.org/pdf/2005.14165.pdf" pdfs/gpt3_few_shot_learners.pdf
fetch "https://arxiv.org/pdf/1810.04805.pdf" pdfs/bert.pdf

echo
echo "Done."
du -h objects/*.jpg ui/*.jpg architecture/*.jpg pdfs/*.pdf 2>/dev/null
