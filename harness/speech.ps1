# Speech -> text with Windows' built-in engines (System.Speech, no installs). Called by harness/speech.py.
#   -Mode file: Input JSON {"phrases": [...], "vocabulary": [...]}. A built-in voice reads each phrase into a WAV,
#               then each WAV is transcribed twice (dictation and domain vocabulary). Writes a JSON list.
#   -Mode mic:  Input JSON {"vocabulary": [...], "timeout_s": n}. Listens once on the default microphone with the
#               domain vocabulary. Writes one JSON object.
# Every transcript comes with the recognizer's confidence (0-1): the lowest of its pieces.
param([string]$Mode, [string]$JobPath, [string]$OutDir, [string]$ResultPath)

Add-Type -AssemblyName System.Speech
$job = Get-Content $JobPath -Raw | ConvertFrom-Json
$culture = New-Object System.Globalization.CultureInfo "en-US"

function New-DomainGrammar {
    # Domain adaptation, the simplest form: the recognizer may ONLY hear these words, 1-40 of them in a row.
    $choices = New-Object System.Speech.Recognition.Choices -ArgumentList (, [string[]]$job.vocabulary)
    $builder = New-Object System.Speech.Recognition.GrammarBuilder
    $builder.Culture = $culture
    $builder.Append($choices, 1, 40)
    New-Object System.Speech.Recognition.Grammar -ArgumentList $builder
}

function Read-Speech($engine, [int]$maxPhrases) {
    # Recognize() returns one phrase at a time (it stops at pauses); at the end of the audio it returns $null
    # or throws "No audio input", depending on where the audio ran out.
    $parts = @(); $confidence = 1.0
    for ($i = 0; $i -lt $maxPhrases; $i++) {
        try { $result = $engine.Recognize() } catch { break }
        if ($null -eq $result) { break }
        $parts += $result.Text
        $confidence = [Math]::Min($confidence, $result.Confidence)
    }
    if ($parts.Count -eq 0) { $confidence = 0.0 }
    [pscustomobject]@{ text = [string]($parts -join " "); confidence = [Math]::Round($confidence, 3) }
}

function Get-Transcript([string]$wav, $grammar) {
    $engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine $culture
    try {
        $engine.LoadGrammar($grammar)
        $engine.SetInputToWaveFile($wav)
        return Read-Speech $engine 10
    } finally {
        $engine.Dispose()
    }
}

if ($Mode -eq "mic") {
    $engine = New-Object System.Speech.Recognition.SpeechRecognitionEngine $culture
    try {
        $engine.LoadGrammar((New-DomainGrammar))
        $engine.SetInputToDefaultAudioDevice()
        $engine.InitialSilenceTimeout = [TimeSpan]::FromSeconds($job.timeout_s)  # how long to wait for speech to start
        $engine.EndSilenceTimeout = [TimeSpan]::FromSeconds(1.0)                # a pause this long ends the command
        $engine.EndSilenceTimeoutAmbiguous = [TimeSpan]::FromSeconds(1.5)
        $heard = Read-Speech $engine 1
    } finally {
        $engine.Dispose()
    }
    ConvertTo-Json $heard | Out-File -Encoding utf8 $ResultPath
    exit
}

New-Item -ItemType Directory -Force $OutDir | Out-Null
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SelectVoice("Microsoft Zira Desktop")
$results = @()
for ($n = 0; $n -lt $job.phrases.Count; $n++) {
    $wav = Join-Path $OutDir "phrase_$n.wav"
    $synth.SetOutputToWaveFile($wav)
    $synth.Speak($job.phrases[$n])
    $synth.SetOutputToNull()
    $results += [pscustomobject]@{
        phrase    = $job.phrases[$n]
        dictation = Get-Transcript $wav (New-Object System.Speech.Recognition.DictationGrammar)
        domain    = Get-Transcript $wav (New-DomainGrammar)
    }
}
$synth.Dispose()
ConvertTo-Json -Depth 3 @($results) | Out-File -Encoding utf8 $ResultPath
