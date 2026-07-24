function Convert-GitDescribeVersion {
    param([string]$Version)

    $Version = $Version -replace '^v', ''
    if ($Version -match "^(\d+)[.](\d+)[.](\d+)-(rc\d+)-(\d+)-(g[0-9a-fA-F]+)(-dirty)?$") {
        return "$($matches[1]).$($matches[2]).$($matches[3])-0.$($matches[4]).$($matches[5]).$($matches[6])$($matches[7] -replace '^-','.')"
    } elseif ($Version -match "^(\d+)[.](\d+)[.](\d+)-(\d+)-(g[0-9a-fA-F]+)(-dirty)?$") {
        return "$($matches[1]).$($matches[2]).$([int]$matches[3] + 1)-0.$($matches[4]).$($matches[5])$($matches[6] -replace '^-','.')"
    }
    return $Version
}
