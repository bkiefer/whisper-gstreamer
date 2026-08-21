getimage() {
    if test -n "$1"; then cd "$1" ; fi
    name=`grep ^name pyproject.toml | head -1 | sed 's/.*"\([^"]*\)".*/\1/'`
    version=`grep ^version pyproject.toml | sed 's/.*"\([^"]*\)".*/\1/'`
    echo -n "$name:$version"
}
