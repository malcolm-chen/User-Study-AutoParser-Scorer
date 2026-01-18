# Transcription Parser

Extracts Q&A pairs from study session transcriptions, scores answers, and exports results to CSV.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=your_key_here
```

Or create a `.env` file:
```
OPENAI_API_KEY=your_key_here
```

## Usage
1. Create three folders at the project root directory: `transcriptions`, `output`, and `extracted_data`
2. Put the transcription files under the `transcriptions` folder
3. Edit the script's main block (`transcription_parser.py`, lines 607-620) with your file paths:

- `TRANSCRIPTION_FILE`: Path to your transcription file (e.g., `.vtt` format)
- `QUESTION_FILE`: Path to the question protocol file (e.g., `Curio 2.0 pilot.txt`)
- `OUTPUT_FILE`: Where to save the CSV output (optional, defaults to same name as transcription with `.csv` extension)
- `MODEL`: OpenAI model to use (default: `gpt-4o`)

Then run:
```bash
python transcription_parser.py
```

## Output

The script generates:
- `output/[filename].csv`: Final results with questions, answers, scores (1-5), and rationales
- `extracted_data/[filename]_extracted.json`: Raw extracted Q&A pairs
- `extracted_data/[filename]_verification.json`: Verification that answers are unedited

CSV columns: `question_id`, `question`, `question_type`, `answer`, `score`, `rationale`

