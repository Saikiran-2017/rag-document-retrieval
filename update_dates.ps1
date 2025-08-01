$workdir = "c:\Users\Sai\Desktop\Main Resume\SAI KIRAN DATA\Data Engineer Master copy\Master AI ML DE Resume\RAG-Based Document Retrieval System\rag-document-retrieval"
Set-Location $workdir

# Create a temporary environment filter script
$filterScript = @'
#!/bin/bash
if [ $GIT_COMMIT_DATE ]
then
 export GIT_AUTHOR_DATE="$(echo $GIT_AUTHOR_DATE | sed 's/2024/2025/')"
 export GIT_COMMITTER_DATE="$(echo $GIT_COMMITTER_DATE | sed 's/2024/2025/')"
fi
'@

# Save the script
$filterScript | Out-File -FilePath "~\env-filter" -Encoding ASCII -NoNewline

Write-Host "Starting git filter-branch to update dates from 2024 to 2025..."

# Run git filter-branch
git filter-branch --env-filter '
if [ $GIT_COMMITTER_DATE ]
then
 export GIT_AUTHOR_DATE="${GIT_AUTHOR_DATE/2024/2025}"
 export GIT_COMMITTER_DATE="${GIT_COMMITTER_DATE/2024/2025}"
fi
' -- --all

Write-Host "Date update complete!"
Write-Host ""
Write-Host "Recent commits with updated dates:"
git log --format="%h %ad %s" --date=short | Select-Object -First 20
