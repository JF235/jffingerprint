param (
    [string]$filename,
    [string]$outputFile = "stdout"
)

Write-Host "Compiling $filename and redirecting output to $outputFile..."

# Compilar o arquivo
g++ $filename -Wall -Wextra -o program -O2

Write-Host "Finished compilation"

# Verificar se a compilação foi bem-sucedida
if ($LASTEXITCODE -ne 0) {
    Write-Host "Compilation failed."
    exit $LASTEXITCODE
}

# Executar o programa e redirecionar a saída
if ($outputFile -eq "stdout") {
    .\program
} else {
    .\program > $outputFile
}
