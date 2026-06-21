import pdfplumber
from pdfminer.high_level import extract_text
import subprocess
import json
import sys
import io
import re
from typing import List, Dict

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def extract_pdftotext(path):
    try:
        result = subprocess.run(["pdftotext", path, "-"], capture_output=True, text=True)
        return result.stdout
    except:
        return ""


# -----------------------------
# TRUE/FALSE EXTRACTOR - DO NOT MODIFY
# -----------------------------
class TFExtractor:
    def extract_true_false(self, text: str) -> List[Dict]:
        questions = []
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            tf_patterns = [
                r'^(.*?)\s+(True|False|true|false)\s*$',
                r'^(True|False|true|false)\s*[\.\:]\s*(.*)$',
                r'^(.*?)\s*[\(\[]\s*(True|False|true|false)\s*[\)\]]\s*$',
                r'^(.*?)\s*(صح|خطأ)\s*$',
            ]
            
            is_tf = False
            question_text = line
            answer = None
            
            for pattern in tf_patterns:
                match = re.match(pattern, line)
                if match:
                    if len(match.groups()) == 2:
                        first_group = match.group(1)
                        second_group = match.group(2)
                        
                        if first_group.lower() in ['true', 'false', 'صح', 'خطأ']:
                            if first_group.lower() in ['true', 'صح']:
                                answer = 'True'
                            else:
                                answer = 'False'
                            question_text = second_group.strip()
                        else:
                            question_text = first_group.strip()
                            if second_group.lower() in ['true', 'صح']:
                                answer = 'True'
                            elif second_group.lower() in ['false', 'خطأ']:
                                answer = 'False'
                    is_tf = True
                    break
            
            if is_tf and question_text:
                question_text = re.sub(r'\s+', ' ', question_text).strip()
                if question_text and len(question_text) > 5:
                    q = {
                        'type': 'true_false',
                        'question': question_text,
                        'options': ['True', 'False'],
                        'answer': answer if answer else 'True'
                    }
                    questions.append(q)
            
            i += 1
        
        return questions


# -----------------------------
# MCQ EXTRACTOR - FIXED WITH BETTER ANSWER DETECTION
# -----------------------------
class MCQExtractor:
    def __init__(self):
        self.raw_text = ""
        
    def extract_from_pdf(self, path: str) -> str:
        text = ""
        
        try:
            with pdfplumber.open(path) as pdf:
                pages = []
                for page in pdf.pages:
                    t = page.extract_text()
                    if t:
                        pages.append(t)
                text = "\n".join(pages)
        except Exception as e:
            print(f"pdfplumber error: {e}", file=sys.stderr)
        
        if not text.strip():
            try:
                text = extract_text(path)
            except Exception as e:
                print(f"pdfminer error: {e}", file=sys.stderr)
        
        if not text.strip():
            text = extract_pdftotext(path)
        
        self.raw_text = text
        return text
    
    def _is_option_line(self, line: str) -> bool:
        # Check for standard option format: A. text, B. text, etc.
        if re.match(r'^\s*[a-dA-D]\s*[\.\)\:]\s*\S', line):
            return True
        # Check for "A. True" or "B. False" in true/false options
        if re.match(r'^\s*[a-dA-D]\s*[\.\)\:]\s*(True|False|true|false)', line):
            return True
        return False
    
    def _get_option_letter(self, line: str) -> str:
        match = re.match(r'^\s*([a-dA-D])\s*[\.\)\:]', line)
        if match:
            return match.group(1).upper()
        return None
    
    def _get_option_text(self, line: str) -> str:
        # Remove the option letter and any trailing whitespace
        text = re.sub(r'^\s*[a-dA-D]\s*[\.\)\:]\s*', '', line)
        return text.strip()
    
    def _extract_answer_from_line(self, line: str) -> str:
        """Extract answer from a line with various formats"""
        if not line:
            return None
            
        clean = line.strip()
        
        # Pattern 1: Answer: B, Correct Answer: B, Ans: B
        patterns = [
            r'(?:answer|ans|correct\s+answer|key)\s*[:=]\s*([a-dA-D])',
            r'(?:answer|ans|correct\s+answer|key)\s+is\s+([a-dA-D])',
            r'^([a-dA-D])\s*[:=]\s*(?:answer|correct)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clean, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Pattern 2: Just a letter (A, B, C, D) - but only if it's standalone
        if re.match(r'^[A-D]\s*$', clean):
            return clean
        
        # Pattern 3: (B) or [B]
        match = re.search(r'[\(\[][\s]*([a-dA-D])[\s]*[\)\]]', clean)
        if match:
            return match.group(1).upper()
        
        # Pattern 4: "The answer is B" or "Correct option is B"
        match = re.search(r'(?:answer|option)\s+(?:is|=)\s+([a-dA-D])', clean, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        
        return None
    
    def _is_table_header(self, line: str) -> bool:
        headers = [
            r'^no\s+question\s+answer',
            r'^question\s+answer',
            r'^no\s+question',
            r'^#\s+question',
            r'^#\s+question\s+answer',
            r'^question\s+type'
        ]
        return any(re.match(h, line, re.IGNORECASE) for h in headers)
    
    def _is_section_break(self, line: str) -> bool:
        breaks = [
            r'match\s+the', r'give\s+the\s+scientific', r'complete\s+the\s+sentence',
            r'answer\s+the\s+following', r'put\s*["\']?\s*true', r'true\s*/\s*false',
            r'صح\s*أم\s*خطأ', r'ضع\s*صح', r'رتب', r'أكمل',
            r'dear learner: put', r'dear learner: answer',
        ]
        return any(re.search(b, line, re.IGNORECASE) for b in breaks)
    
    def _is_question_text(self, line: str) -> bool:
        if not line: 
            return False
        if self._is_option_line(line): 
            return False
        if self._extract_answer_from_line(line): 
            return False
        if self._is_table_header(line): 
            return False
        if self._is_section_break(line): 
            return False
        if len(line) < 5: 
            return False
        
        # Check if line starts with a number (like "41 What is Laravel?")
        if re.match(r'^\s*\d+\s+', line):
            rest = re.sub(r'^\s*\d+\s+', '', line)
            if len(rest) > 5:
                return True
        
        # Check for question patterns
        if re.match(r'^[A-Z\?]|^(what|which|who|when|where|why|how|the|a|an|in|on|if|to)\s', line, re.IGNORECASE):
            return True
        
        # Check if line ends with a question mark (likely a question)
        if re.search(r'\?$', line):
            return True
        
        return False
    
    def _clean_question_text(self, line: str) -> str:
        """Remove question numbers, answer letters, and clean the text"""
        # Remove leading numbers like "41 " or "1. "
        cleaned = re.sub(r'^\s*\d+\s*[\.\:\)]?\s*', '', line)
        # Remove "No. Question Answer" pattern
        cleaned = re.sub(r'^No\.\s+Question\s+Answer\s+', '', cleaned, flags=re.IGNORECASE)
        # Remove "Q" prefix
        cleaned = re.sub(r'^Q\d+\s*[\.\:\)]\s*', '', cleaned, flags=re.IGNORECASE)
        
        # CRITICAL FIX: Remove trailing answer letters (a, b, c, d)
        # This matches "What is PHP? b" -> "What is PHP?"
        cleaned = re.sub(r'\s+[a-dA-D]\s*$', '', cleaned)
        # Remove answer in parentheses: (b) or [b]
        cleaned = re.sub(r'\s*[\(\[]\s*[a-dA-D]\s*[\)\]]\s*$', '', cleaned)
        # Remove answer with dot: "b." at the end
        cleaned = re.sub(r'\s+[a-dA-D]\.\s*$', '', cleaned)
        
        return cleaned.strip()
    
    def _is_page_number(self, line: str) -> bool:
        clean = re.sub(r'[^\x20-\x7E]', '', line).strip()
        return bool(re.match(r'^\d{1,3}$', clean))
    
    def _clean(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()
    
    def extract_mcq_questions(self, text: str) -> List[Dict]:
        questions = []
        lines = text.split('\n')
        lines = [l.strip() for l in lines]
        
        i = 0
        n = len(lines)
        
        while i < n:
            line = lines[i]
            
            # Skip headers, page numbers, and empty lines
            if (self._is_table_header(line) or 
                self._is_section_break(line) or 
                self._is_page_number(line) or
                not line):
                i += 1
                continue
            
            # Check if this is a question line
            if self._is_question_text(line):
                # Extract answer letter from the original line
                answer_letter = None
                # Check for trailing letter (a, b, c, d) at the end of the line
                trailing_match = re.search(r'\s+([a-dA-D])\s*$', line)
                if trailing_match:
                    answer_letter = trailing_match.group(1).upper()
                
                question_text = line
                # Clean the question text (removes numbers and answer letters)
                question_text = self._clean_question_text(question_text)
                i += 1
                
                # Collect multi-line question if needed
                while i < n:
                    next_line = lines[i]
                    # Stop if we hit options or another question
                    if self._is_option_line(next_line):
                        break
                    if self._is_question_text(next_line):
                        # If next line looks like a question, it might be a continuation
                        if len(next_line) < 30 or re.match(r'^(what|which|who|when|where|why|how|the|a)\s', next_line, re.IGNORECASE):
                            # Check if this continuation has a trailing answer letter
                            trailing_match2 = re.search(r'\s+([a-dA-D])\s*$', next_line)
                            if trailing_match2 and not answer_letter:
                                answer_letter = trailing_match2.group(1).upper()
                            cleaned_next = self._clean_question_text(next_line)
                            question_text += ' ' + cleaned_next
                            i += 1
                            continue
                        break
                    if self._is_section_break(next_line):
                        break
                    if self._is_table_header(next_line):
                        break
                    
                    # Add non-option lines to question text
                    cleaned_next = self._clean_question_text(next_line)
                    if cleaned_next:
                        question_text += ' ' + cleaned_next
                    i += 1
                
                question_text = self._clean(question_text)
                
                # Collect options
                options = []
                option_letters = []
                
                while i < n:
                    next_line = lines[i]
                    
                    # Check for answer in this line
                    ans = self._extract_answer_from_line(next_line)
                    if ans:
                        answer_letter = ans
                        i += 1
                        continue
                    
                    # Check if this is an option line
                    if self._is_option_line(next_line):
                        letter = self._get_option_letter(next_line)
                        opt_text = self._get_option_text(next_line)
                        if opt_text:
                            options.append(opt_text)
                            option_letters.append(letter)
                        i += 1
                        continue
                    
                    # Check if we've hit the next question or section break
                    if self._is_question_text(next_line):
                        break
                    if self._is_section_break(next_line):
                        break
                    if self._is_table_header(next_line):
                        break
                    
                    # If we have options and this line looks like continuation text
                    if options and next_line:
                        if options and len(options) > 0:
                            options[-1] += ' ' + next_line
                        i += 1
                        continue
                    
                    i += 1
                
                # Clean up options
                final_options = []
                final_letters = []
                seen_texts = set()
                
                for j, o in enumerate(options):
                    clean_o = self._clean(o)
                    if clean_o and len(clean_o) > 1 and clean_o.lower() not in seen_texts:
                        final_options.append(clean_o)
                        if j < len(option_letters):
                            final_letters.append(option_letters[j])
                        else:
                            final_letters.append(chr(65 + j))
                        seen_texts.add(clean_o.lower())
                
                options = final_options
                option_letters = final_letters
                
                # Determine if this is MCQ or True/False
                is_tf = False
                if options:
                    tf_values = ['true', 'false', 'صح', 'خطأ']
                    tf_count = sum(1 for o in options if o.lower() in tf_values)
                    if tf_count == 2 and len(options) == 2:
                        is_tf = True
                
                # Only add if we have a valid question with options
                if question_text and len(options) >= 2:
                    # Final cleanup - remove any remaining trailing letters
                    question_text = re.sub(r'\s+[a-dA-D]\s*$', '', question_text)
                    question_text = re.sub(r'\s*[\(\[]\s*[a-dA-D]\s*[\)\]]\s*$', '', question_text)
                    question_text = re.sub(r'\s+[a-dA-D]\.\s*$', '', question_text)
                    question_text = self._clean(question_text)
                    
                    if is_tf:
                        q = {
                            'type': 'true_false',
                            'question': question_text,
                            'options': ['True', 'False'],
                            'answer': answer_letter if answer_letter else None
                        }
                        questions.append(q)
                    else:
                        q = {
                            'type': 'multiple_choice',
                            'question': question_text,
                            'options': options
                        }
                        
                        # Set answer if found
                        if answer_letter and answer_letter in option_letters:
                            q['answer'] = answer_letter
                        else:
                            # Try to infer answer from option matching
                            # If no answer found, set to None (will be flagged as unverified)
                            q['answer'] = None
                        
                        questions.append(q)
            else:
                i += 1
        
        return questions


# -----------------------------
# Main PDF extractor
# -----------------------------
def extract_pdf(path):
    try:
        mcq_extractor = MCQExtractor()
        text = mcq_extractor.extract_from_pdf(path)
        
        if not text.strip():
            return {"success": False, "error": "PDF has no extractable text.", "text": "", "questions": []}
        
        mcq_questions = mcq_extractor.extract_mcq_questions(text)

        # RESTORED: the dedicated True/False pass. TFExtractor itself was
        # never broken — this call to it was simply removed at some point
        # when extract_mcq_questions was rewritten, which is why True/False
        # detection stopped working even though the class above is intact.
        tf_extractor = TFExtractor()
        tf_questions = tf_extractor.extract_true_false(text)

        all_questions = mcq_questions + tf_questions
        
        # Remove duplicates
        seen = set()
        unique = []
        for q in all_questions:
            qt = q['question'].lower().strip()
            if qt not in seen:
                seen.add(qt)
                unique.append(q)
        
        if not unique:
            return {"success": False, "error": "No questions could be extracted.", "text": text[:2000], "questions": []}
        
        return {"success": True, "text": text[:1000], "questions": unique}
        
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        return {"success": False, "error": str(e), "text": "", "questions": []}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No file path", "text": "", "questions": []}, ensure_ascii=False))
        sys.exit(1)
    
    result = extract_pdf(sys.argv[1])
    print(json.dumps(result, ensure_ascii=False))