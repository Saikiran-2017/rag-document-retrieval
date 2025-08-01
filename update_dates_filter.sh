#!/bin/bash
# Update commit dates from 2024 to 2025
if [ -n "$GIT_AUTHOR_DATE" ]; then
  export GIT_AUTHOR_DATE="${GIT_AUTHOR_DATE//2024/2025}"
fi
if [ -n "$GIT_COMMITTER_DATE" ]; then
  export GIT_COMMITTER_DATE="${GIT_COMMITTER_DATE//2024/2025}"
fi
