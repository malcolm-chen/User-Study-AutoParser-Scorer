#!/usr/bin/env python3
"""
LLM-based Transcription Parser for User Study Sessions
Extracts pre-test and post-test Q&A pairs from transcriptions and scores answers.
"""

import os
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class TranscriptionParser:
    """Parse transcription files to extract Q&A pairs using LLM."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        """
        Initialize the parser.
        
        Args:
            api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
            model: Model to use for parsing (default: gpt-4o)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
    
    def get_temperature(self):
        """Get temperature parameter based on model. GPT-5 only supports default (1)."""
        if self.model.startswith("gpt-5"):
            return None  # Use default (1) for GPT-5
        return 0.1  # Use 0.1 for other models
    
    def get_reasoning_effort(self):
        """Get reasoning_effort parameter based on model. GPT-5 supports low/medium/high."""
        if self.model.startswith("gpt-5"):
            return "low"  # Use low effort for GPT-5 extraction
        return None  # Not applicable for other models
    
    def load_text_file(self, file_path: str) -> str:
        """Load text from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def parse_and_extract_all(self, question_file: str, transcription: str) -> Dict[str, Any]:
        """
        Combined method: Parse question protocol file and extract Q&A from transcription in one LLM inference.
        Returns a dict with protocol questions, correct answers, and extracted Q&A pairs.
        """
        question_content = self.load_text_file(question_file)
        
        system_prompt = """You are an expert at parsing research study documents and transcriptions. You will be provided with:
1. A user study protocol document containing pre-test and post-test questions with correct answers
2. A transcription of the actual study session

Your task is to:
1. Parse the protocol document to extract all pre-test and post-test questions along with their correct answers
2. Identify when the researcher asks questions (including main questions and follow-up questions) in the transcription
3. Extract the participant's answers, merging any answers that are split across multiple timestamp segments

<Instructions>
- First, understand the protocol: Extract all pre-test and post-test questions from the protocol document, noting their question numbers and correct answers.
- Then, based on the protocol, parse the transcription: Identify the role of different speakers (typically researcher and participant).
- Extract BOTH the question text that was actually asked by the researcher AND the participant's answer.
- The researcher may ask follow-up questions. Classify each as either "main" (matches a protocol question) or "follow-up" (additional questions).
- IMPORTANT: Participant answers may be split across multiple timestamp segments (e.g., in VTT format). You MUST merge all parts of a single answer. For example:
  - "00:47:14.430 --> 00:47:18.970\nAanav: Because…"
  - "00:47:20.300 --> 00:47:32.259\nAanav: The girl… the girl in the left, maybe didn't rub it to the sofa, but the girl in the right, rubbed it to the sofa."
  Should become: "Because… The girl… the girl in the left, maybe didn't rub it to the sofa, but the girl in the right, rubbed it to the sofa."
- Match extracted questions to protocol question numbers. Follow-ups use the same question_number as the main question.
- IMPORTANT: DO NOT MODIFY THE TRANSCRIPTION. Extract exact question and answer text as spoken.
</Instructions>

<Output Format>
Return a JSON structure:
{{
  "pre_test": [
      {{"question_number": 1, "question_type": "main", "question_text": "...", "answer": "..."}},
      {{"question_number": 1, "question_type": "follow-up", "question_text": "...", "answer": "..."}},
      ...
    ],
    "post_test": [
      {{"question_number": 1, "question_type": "main", "question_text": "...", "answer": "..."}},
      {{"question_number": 1, "question_type": "follow-up", "question_text": "...", "answer": "..."}},
      ...
    ]
}}

For extracted Q&A:
- "question_number": Match to protocol question number
- "question_type": "main" or "follow-up"
- "question_text": Exact question text from transcription
- "answer": Complete merged answer from participant

Do not include any other text in your response.
</Output Format>
"""
        
        user_prompt = f"""Parse the protocol document and extract questions and answers from the transcription:

<PROTOCOL DOCUMENT>
{question_content}
</PROTOCOL DOCUMENT>

<TRANSCRIPTION>
{transcription}
</TRANSCRIPTION>

Return only valid JSON with the structure specified above."""
        
        try:
            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            # Only include temperature if not using default
            temp = self.get_temperature()
            if temp is not None:
                request_params["temperature"] = temp
            
            # Add reasoning_effort for GPT-5 (low effort for extraction)
            reasoning_effort = self.get_reasoning_effort()
            if reasoning_effort is not None:
                request_params["reasoning_effort"] = reasoning_effort
            
            response = self.client.chat.completions.create(**request_params)
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from markdown if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            return json.loads(result_text)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response was: {result_text}")
            raise
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise
    
    def score_all_answers(self, extracted_qa: Dict[str, Any], protocol_file: str) -> Dict[str, Any]:
        """
        Score all answers in one inference using the entire extracted content and protocol.
        Returns a dict mapping question identifiers to scores and rationales.
        """
        protocol_content = self.load_text_file(protocol_file)
        
        # Convert extracted Q&A to JSON string for the prompt
        extracted_json = json.dumps(extracted_qa, indent=2, ensure_ascii=False)
        
        system_prompt = """You are a children's education expert at scoring children's answers to a study session where the child interacted with a system that was designed to help them explore science concepts. 

You will be provided with:
1. The study protocol document containing pre-test and post-test questions with correct answers
2. The extracted questions and answers from the transcription

Your task is to score all answers based on the protocol's correct answers using a 5-point Likert scale.

<Scoring Guidelines>
Use a 5-point Likert scale (1-5) to evaluate each answer:
- 5 (Excellent): Answer is completely correct and demonstrates deep understanding of the concept. Shows clear grasp of the underlying principles.
- 4 (Good): Answer is mostly correct with minor gaps or incomplete explanations. Demonstrates solid understanding of the concept.
- 3 (Moderate): Answer shows partial understanding but contains some inaccuracies or significant gaps. Demonstrates basic grasp but may miss key aspects.
- 2 (Poor): Answer shows limited understanding with major inaccuracies or misconceptions. May have some correct elements but overall understanding is weak.
- 1 (Very Poor): Answer is largely incorrect or demonstrates significant misunderstanding. Shows minimal to no understanding of the concept.

Consider that children may express correct ideas in different ways. Be generous but accurate in your assessment.
For each answer, provide a brief rationale (1-2 sentences) explaining your scoring decision.
</Scoring Guidelines>

<Output Format>
Return a JSON structure matching the input structure, but with scores and rationales added:
{{
  "pre_test": [
    {{
      "question_number": 1,
      "question_type": "main",
      "question_text": "...",
      "answer": "...",
      "score": 1-5,
      "rationale": "brief explanation"
    }},
    ...
  ],
  "post_test": [
    {{
      "question_number": 1,
      "question_type": "main",
      "question_text": "...",
      "answer": "...",
      "score": 1-5,
      "rationale": "brief explanation"
    }},
    ...
  ]
}}

Match each extracted Q&A item to the corresponding protocol question using question_number. If no answer is provided (empty string), score as 1 with rationale "No answer provided".
</Output Format>
"""
        
        user_prompt = f"""Please score all the extracted answers based on the study protocol:

<STUDY PROTOCOL>
{protocol_content}
</STUDY PROTOCOL>

<EXTRACTED QUESTIONS AND ANSWERS>
{extracted_json}
</EXTRACTED QUESTIONS AND ANSWERS>

Return only valid JSON with the structure specified above, including scores and rationales for all answers."""
        
        try:
            # Build request parameters
            request_params = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            # Only include temperature if not using default
            temp = self.get_temperature()
            if temp is not None:
                request_params["temperature"] = temp
            
            response = self.client.chat.completions.create(**request_params)
            
            result_text = response.choices[0].message.content.strip()
            
            # Extract JSON from markdown if present
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0].strip()
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(result_text)
            return result
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON response: {e}")
            print(f"Response was: {result_text}")
            raise
        except Exception as e:
            print(f"Error scoring answers: {e}")
            raise
    
    def process_with_scoring(self, transcription: str, question_file: str, transcription_file: str = None) -> List[Dict]:
        """
        Main processing function: parse protocol and extract Q&A in one inference, then score answers.
        Returns list of dicts with question_id, question, answer, score, rationale.
        
        Args:
            transcription: Transcription content as string
            question_file: Path to question protocol file
            transcription_file: Path to transcription file (used for naming intermediate JSON)
        """
        # Parse protocol and extract Q&A in one LLM inference
        print("Parsing protocol and extracting questions and answers from transcription...")
        combined_result = self.parse_and_extract_all(question_file, transcription)
        
        # Save intermediate extraction results
        from pathlib import Path
        
        # Create extracted_data folder if it doesn't exist
        extracted_data_dir = Path("extracted_data")
        extracted_data_dir.mkdir(exist_ok=True)
        
        # Use transcription file name for the intermediate JSON file
        if transcription_file:
            base_name = Path(transcription_file).stem
        else:
            base_name = "extracted"
        
        intermediate_file = extracted_data_dir / f"{base_name}_extracted.json"
        self.save_intermediate_json(combined_result, str(intermediate_file))
        
        # Get extracted Q&A (combined_result already has pre_test and post_test at root level)
        extracted_qa = {
            'pre_test': combined_result.get('pre_test', []),
            'post_test': combined_result.get('post_test', [])
        }
        
        # Verify all answers are unedited
        print("Verifying extracted answers are unedited...")
        verification_results = self.verify_all_answers(extracted_qa, transcription)
        
        # Save verification results
        verification_file = extracted_data_dir / f"{base_name}_verification.json"
        self.save_intermediate_json(verification_results, str(verification_file))
        
        # Print verification summary
        summary = verification_results['summary']
        print(f"Verification: {summary['verified']}/{summary['total']} answers verified "
              f"({summary['verification_rate']:.1f}%)")
        if summary['not_verified'] > 0:
            print(f"Warning: {summary['not_verified']} answers could not be verified as unedited")
        
        # Score all answers in one inference
        print("Scoring all answers...")
        scored_results = self.score_all_answers(extracted_qa, question_file)
        
        # Compile results from scored output
        results = []
        
        # Process pre-test
        for item in scored_results.get('pre_test', []):
            q_num = item.get('question_number', '')
            q_type = item.get('question_type', 'main')
            # Create question ID: pre-test_Q{number}_{type}
            question_id = f"pre-test_Q{q_num}_{q_type}" if q_num else f"pre-test_{q_type}"
            results.append({
                'question_id': question_id,
                'question': item.get('question_text', ''),
                'question_type': q_type,
                'answer': item.get('answer', ''),
                'score': item.get('score', 1),
                'rationale': item.get('rationale', '')
            })
        
        # Process post-test
        for item in scored_results.get('post_test', []):
            q_num = item.get('question_number', '')
            q_type = item.get('question_type', 'main')
            # Create question ID: post-test_Q{number}_{type}
            question_id = f"post-test_Q{q_num}_{q_type}" if q_num else f"post-test_{q_type}"
            results.append({
                'question_id': question_id,
                'question': item.get('question_text', ''),
                'question_type': q_type,
                'answer': item.get('answer', ''),
                'score': item.get('score', 1),
                'rationale': item.get('rationale', '')
            })
        
        return results
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison by:
        - Removing extra whitespace (multiple spaces, tabs, newlines)
        - Converting to lowercase for case-insensitive matching
        - Removing punctuation for fuzzy matching
        """
        if not text:
            return ""
        # Normalize whitespace: replace all whitespace with single space
        normalized = re.sub(r'\s+', ' ', text.strip())
        return normalized
    
    def verify_answer_unedited(self, extracted_answer: str, transcription: str, 
                               question_text: str = "") -> Dict[str, Any]:
        """
        Verify that the extracted answer appears unedited in the original transcription.
        
        Args:
            extracted_answer: The answer text extracted by the LLM
            transcription: The original transcription text
            question_text: Optional question text for context
        
        Returns:
            Dict with verification results:
            - verified: bool - Whether answer is verified as unedited
            - match_type: str - Type of match found ('exact', 'normalized', 'substring', 'fuzzy', 'not_found')
            - confidence: float - Confidence score (0.0-1.0)
            - details: str - Additional details about the verification
            - found_text: str - The actual text found in transcription (if any)
        """
        if not extracted_answer or not extracted_answer.strip():
            return {
                'verified': True,  # Empty answers are considered valid
                'match_type': 'empty',
                'confidence': 1.0,
                'details': 'Empty answer - no verification needed',
                'found_text': ''
            }
        
        # Normalize both texts for comparison
        normalized_answer = self.normalize_text(extracted_answer)
        normalized_transcription = self.normalize_text(transcription)
        
        # Rule 1: Exact match (case-insensitive, normalized whitespace)
        if normalized_answer.lower() in normalized_transcription.lower():
            # Try to find the exact location
            answer_lower = normalized_answer.lower()
            trans_lower = normalized_transcription.lower()
            idx = trans_lower.find(answer_lower)
            if idx != -1:
                found_text = normalized_transcription[idx:idx+len(normalized_answer)]
                return {
                    'verified': True,
                    'match_type': 'normalized',
                    'confidence': 1.0,
                    'details': 'Answer found in transcription with normalized whitespace',
                    'found_text': found_text
                }
        
        # Rule 2: Check if answer is a substring (for answers split across timestamps)
        # Remove all whitespace and compare
        answer_no_space = re.sub(r'\s+', '', extracted_answer.lower())
        trans_no_space = re.sub(r'\s+', '', transcription.lower())
        
        if answer_no_space in trans_no_space:
            return {
                'verified': True,
                'match_type': 'substring',
                'confidence': 0.9,
                'details': 'Answer found as substring (may be split across timestamps)',
                'found_text': extracted_answer
            }
        
        # Rule 3: Check for key phrases/words from the answer
        # Extract significant words (3+ characters, excluding common words)
        answer_words = [w.lower() for w in re.findall(r'\b\w{3,}\b', extracted_answer)]
        if answer_words:
            matching_words = sum(1 for word in answer_words if word in transcription.lower())
            word_match_ratio = matching_words / len(answer_words) if answer_words else 0
            
            if word_match_ratio >= 0.7:  # At least 70% of significant words match
                return {
                    'verified': True,
                    'match_type': 'fuzzy',
                    'confidence': word_match_ratio,
                    'details': f'Found {matching_words}/{len(answer_words)} key words in transcription',
                    'found_text': extracted_answer
                }
        
        # Rule 4: Check if answer contains only punctuation/formatting differences
        # Remove punctuation and compare
        answer_no_punct = re.sub(r'[^\w\s]', '', extracted_answer.lower())
        trans_no_punct = re.sub(r'[^\w\s]', '', transcription.lower())
        answer_no_punct_clean = re.sub(r'\s+', ' ', answer_no_punct).strip()
        trans_no_punct_clean = re.sub(r'\s+', ' ', trans_no_punct).strip()
        
        if answer_no_punct_clean and answer_no_punct_clean in trans_no_punct_clean:
            return {
                'verified': True,
                'match_type': 'fuzzy',
                'confidence': 0.8,
                'details': 'Answer found with punctuation differences only',
                'found_text': extracted_answer
            }
        
        # Not verified - answer not found in transcription
        return {
            'verified': False,
            'match_type': 'not_found',
            'confidence': 0.0,
            'details': 'Answer text not found in original transcription - may have been edited',
            'found_text': ''
        }
    
    def verify_all_answers(self, extracted_qa: Dict[str, Any], transcription: str) -> Dict[str, Any]:
        """
        Verify all extracted answers against the original transcription.
        
        Returns:
            Dict with verification results for each answer, keyed by question_id
        """
        verification_results = {
            'pre_test': [],
            'post_test': [],
            'summary': {
                'total': 0,
                'verified': 0,
                'not_verified': 0,
                'verification_rate': 0.0
            }
        }
        
        total = 0
        verified = 0
        
        # Verify pre-test answers
        for item in extracted_qa.get('pre_test', []):
            q_num = item.get('question_number', '')
            q_type = item.get('question_type', 'main')
            question_id = f"pre-test_Q{q_num}_{q_type}" if q_num else f"pre-test_{q_type}"
            answer = item.get('answer', '')
            question_text = item.get('question_text', '')
            
            verification = self.verify_answer_unedited(answer, transcription, question_text)
            verification['question_id'] = question_id
            verification['question_number'] = q_num
            verification['question_type'] = q_type
            
            verification_results['pre_test'].append(verification)
            total += 1
            if verification['verified']:
                verified += 1
        
        # Verify post-test answers
        for item in extracted_qa.get('post_test', []):
            q_num = item.get('question_number', '')
            q_type = item.get('question_type', 'main')
            question_id = f"post-test_Q{q_num}_{q_type}" if q_num else f"post-test_{q_type}"
            answer = item.get('answer', '')
            question_text = item.get('question_text', '')
            
            verification = self.verify_answer_unedited(answer, transcription, question_text)
            verification['question_id'] = question_id
            verification['question_number'] = q_num
            verification['question_type'] = q_type
            
            verification_results['post_test'].append(verification)
            total += 1
            if verification['verified']:
                verified += 1
        
        # Calculate summary
        verification_results['summary'] = {
            'total': total,
            'verified': verified,
            'not_verified': total - verified,
            'verification_rate': (verified / total * 100) if total > 0 else 0.0
        }
        
        return verification_results
    
    def save_intermediate_json(self, data: Dict[str, Any], output_path: str):
        """Save intermediate extracted Q&A data to JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Intermediate extraction results saved to {output_path}")
    
    def save_to_csv(self, results: List[Dict], output_path: str):
        """Save results to CSV file with columns: question_id, question, question_type, answer, score, rationale."""
        # Use utf-8-sig encoding to include BOM for proper UTF-8 recognition in Excel
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header with question_id as first column
            writer.writerow(['question_id', 'question', 'question_type', 'answer', 'score', 'rationale'])
            
            # Write data
            for item in results:
                writer.writerow([
                    item.get('question_id', ''),
                    item.get('question', ''),
                    item.get('question_type', 'main'),
                    item.get('answer', ''),
                    item.get('score', ''),
                    item.get('rationale', '')
                ])
        
        print(f"Results saved to {output_path}")


def process_transcription(transcription_file: str, question_file: str, 
                         output_file: str = None, api_key: str = None, 
                         model: str = "gpt-4o"):
    """
    Main function to process a transcription file and generate scored CSV output.
    
    Args:
        transcription_file: Path to the transcription text file
        question_file: Path to text file containing pre-test and post-test questions
        output_file: Output CSV file path (default: transcription_file with .csv extension)
        api_key: OpenAI API key (if not provided, uses OPENAI_API_KEY env var)
        model: OpenAI model to use (default: gpt-4o)
    """
    # Set default output file if not provided
    if output_file is None:
        output_file = Path(transcription_file).with_suffix('.csv')
    
    # Initialize parser
    parser_instance = TranscriptionParser(api_key=api_key, model=model)
    
    # Load transcription
    print(f"Loading transcription from {transcription_file}...")
    transcription = parser_instance.load_text_file(transcription_file)
    
    # Process transcription with scoring
    results = parser_instance.process_with_scoring(
        transcription, question_file, transcription_file
    )
    
    # Save to CSV
    print(f"Saving results to {output_file}...")
    parser_instance.save_to_csv(results, output_file)
    
    print("Done!")
    return results


# Example usage:
if __name__ == "__main__":
    # Set your parameters here
    TRANSCRIPTION_FILE = "transcriptions/S09_Transcript.vtt"  # Path to your transcription file
    QUESTION_FILE = "Curio 2.0 pilot.txt"  # Path to your question file
    OUTPUT_FILE = "output/S09_Transcript.csv"  # Path for output CSV file (optional, defaults to transcription_file.csv)ssed
    API_KEY = None  # Set to your API key, or None to use OPENAI_API_KEY env var
    MODEL = "gpt-5"  # Model to use
    
    # Process the transcription
    process_transcription(
        transcription_file=TRANSCRIPTION_FILE,
        question_file=QUESTION_FILE,
        output_file=OUTPUT_FILE,
        api_key=API_KEY,
        model=MODEL
    )
