# Exports docs/presentation/BTP_endterm.pptx to a PDF and to one PNG per slide with Microsoft PowerPoint.
#   powershell -ExecutionPolicy Bypass -File docs/presentation/export_with_powerpoint.ps1 [-OutDir <folder for PNGs>]
param([string]$OutDir = "")
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$pptx = Join-Path $here "BTP_endterm.pptx"
$pdf = Join-Path $here "BTP_endterm.pdf"
if ($OutDir -eq "") { $OutDir = Join-Path $here "render" }
New-Item -ItemType Directory -Force $OutDir | Out-Null
Get-ChildItem $OutDir -Filter "*.png" -ErrorAction SilentlyContinue | Remove-Item -Force

$app = New-Object -ComObject PowerPoint.Application
try {
    # Open(FileName, ReadOnly, Untitled, WithWindow)
    $p = $app.Presentations.Open($pptx, -1, 0, 0)
    $p.SaveAs($pdf, 32)                      # 32 = ppSaveAsPDF
    $i = 1
    foreach ($s in $p.Slides) {
        $s.Export((Join-Path $OutDir ("slide-{0:D2}.png" -f $i)), "PNG", 1600, 900)
        $i++
    }
    $p.Close()
    Write-Output "PDF: $pdf"
    Write-Output "PNGs: $OutDir ($($i - 1) slides)"
} finally {
    $app.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
