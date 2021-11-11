# set desired path and output path
$desired_path="./*.wav"
$out_path="./out"

$list_total = get-childitem $desired_path -Name

$counter = [pscustomobject] @{ Value = 0 }
$arr_size = 100
$batch = $list_total | Group-Object -Property { [math]::Floor($counter.Value++ / $arr_size) }

foreach ($f in $batch) {
    spleeter separate -o $out_path -p spleeter:2stems-16kHz -f "{filename}_{instrument}.{codec}" $f.Group
}

Get-ChildItem $out_path -filter *_vocals.wav | Move-Item -Destination "$out_path/vocals"
Get-ChildItem $out_path -filter *_accompanimnet.wav | Move-Item -Destination "$out_path/accompanimnet"

# rename to exclude stem info
Get-ChildItem "$out_path./vocals" | rename-item -newname {$_.name -replace "_vocals", ""}
Get-ChildItem "$out_path./accompanimnet" | rename-item -newname {$_.name -replace "_accompanimnet", ""}