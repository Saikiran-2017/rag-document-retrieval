$workdir = "c:\Users\Sai\Desktop\Main Resume\SAI KIRAN DATA\Data Engineer Master copy\Master AI ML DE Resume\RAG-Based Document Retrieval System\rag-document-retrieval"
Set-Location $workdir

Write-Host "Extracting commit information..."

# Get all commits in reverse order (oldest first)
$commits = git log --reverse --format="%H %ad %s" --date=iso-strict 

# Store in array
$commitList = @()
foreach ($line in $commits) {
    $parts = $line -split ' ', 3
    if ($parts.Count -ge 3) {
        $hash = $parts[0]
        $date = $parts[1]
        $time = $parts[2]
        $msg = $line.Substring($hash.Length + $date.Length + $time.Length + 3)
        
        # Convert 2024 to 2025
        $date = $date -replace '(2024)', '2025'
        
        $commitList += @{
            hash = $hash
            date = $date
            time = $time
            message = $msg
        }
    }
}

Write-Host "Found $($commitList.Count) commits"
Write-Host "First commit: $($commitList[0].message) on $($commitList[0].date)"
Write-Host "Last commit: $($commitList[-1].message) on $($commitList[-1].date)"
