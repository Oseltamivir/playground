#!/bin/zsh

# Directory to read (default to current dir if none given)
DIR="${1:-.}"

# Find all regular files (non-recursive; add -type f to restrict to files only)
# Use -type f for only files; remove it if you want directories as well.
find "$DIR" -type f | while IFS= read -r file; do
    echo "$file"
    echo '```'
    cat "$file"
    echo '```'
    echo    # Blank line between files
done

