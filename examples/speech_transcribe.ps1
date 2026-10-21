# Text -> speech (WAV) -> text, using only Windows' built-in engines (System.Speech, no installs).
# Called by 07_speech.py. Input JSON: {"phrases": [...], "vocabulary": [...]}. Writes results JSON.
param([string]$JobPath, [string]$OutDir, [string]$ResultPath)

Add-Type -AssemblyName System.Speech
$job = Get-Content $JobPath -Raw | ConvertFrom-Json
New-Item -ItemType Directory -Force $OutDir | Out-Null

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice("Microsoft Zira Desktop")
$culture = New-Object System.Globalization.CultureInfo "en-US"

function New-DomainGrammar {
    # Domain adaptation, the simplest form: the recognizer may ONLY hear these words, 1-40 of them in a row.
    $choices = New-Object System.Speech.Recognition.Choices -ArgumentList (, [string[]]$job.vocabulary)
    $builder = New-Object System.Speech.Recognition.GrammarBuilder
    $builder.Culture = $culture
    $builder.Append($choices, 1, 40)
    New-Object System.Speech.Recognition.Grammar -ArgumentList $builder
}

function Get-Transcript([string]$wav, $grammar) {
    $engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine $culture
    try {
        $engine.LoadGrammar($grammar)
        $engine.SetInputToWaveFile($wav)
        $parts = @()
        # Recognize() returns one phrase at a time (it stops at pauses); at end of file it returns $null
        # or throws "No audio input", depending on where the audio ran out.
        for ($i = 0; $i -lt 10; $i++) {
            try { $result = $engine.Recognize() } catch { break }
            if ($null -eq $result) { break }
            $parts += $result.Text
        }
        return [string]($parts -join " ")
    } finally {
        $engine.Dispose()
    }
}

$results = @()
for ($n = 0; $n -lt $job.phrases.Count; $n++) {
    $phrase = $job.phrases[$n]
    $wav = Join-Path $OutDir "phrase_$n.wav"
    $synth.SetOutputToWaveFile($wav)
    $synth.Speak($phrase)
    $synth.SetOutputToNull()
    $results += [pscustomobject]@{
        phrase    = $phrase
        dictation = [string](Get-Transcript $wav (New-Object System.Speech.Recognition.DictationGrammar))
        domain    = [string](Get-Transcript $wav (New-DomainGrammar))
    }
}
$synth.Dispose()
ConvertTo-Json @($results) | Out-File -Encoding utf8 $ResultPath
