#!/bin/bash
# This script will be run by git filter-branch
# It updates commit dates from 2024 to 2025

export GIT_AUTHOR_DATE="${GIT_AUTHOR_DATE//2024/2025}"
export GIT_COMMITTER_DATE="${GIT_COMMITTER_DATE//2024/2025}"
