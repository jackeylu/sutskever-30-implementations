#!/usr/bin/env python3
"""
Attempt to repair incomplete PDF files
"""

import os
import re
import shutil

PDF_DIR = "D:/codes/learn/pdf/pdf"
BACKUP_DIR = os.path.join(PDF_DIR, "backup")

def repair_pdf_by_adding_eof(filepath):
    """Try to repair PDF by adding EOF marker and xref table"""
    try:
        # Read the file
        with open(filepath, 'rb') as f:
            content = f.read()

        # Check if it's already repaired
        if b'%%EOF' in content:
            return False, "File already has EOF marker"

        # Backup original
        backup_path = os.path.join(BACKUP_DIR, os.path.basename(filepath))
        os.makedirs(BACKUP_DIR, exist_ok=True)
        shutil.copy2(filepath, backup_path)

        # Find the last %%EOF marker if it exists
        last_eof_pos = content.rfind(b'%%EOF')

        if last_eof_pos == -1:
            # No EOF marker found, try to add it
            # Find the last occurrence of some common PDF patterns
            # Look for end of objects or streams
            patterns = [
                b'endobj\n',
                b'endobj\r\n',
                b'endstream\n',
                b'endstream\r\n',
            ]

            last_pos = 0
            for pattern in patterns:
                pos = content.rfind(pattern)
                if pos > last_pos:
                    last_pos = pos + len(pattern)

            if last_pos > 0:
                # Truncate to last valid position and add EOF
                repaired_content = content[:last_pos]
                # Add minimal EOF section
                eof_marker = b'\n%%EOF\n'
                repaired_content += eof_marker

                # Write repaired file
                with open(filepath, 'wb') as f:
                    f.write(repaired_content)

                return True, "Added EOF marker"
            else:
                return False, "Could not find valid end position"

        return False, "EOF marker already exists"

    except Exception as e:
        return False, f"Error: {str(e)}"

def verify_pdf(filepath):
    """Verify if PDF can be opened"""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                return False, "Invalid PDF header"

            # Check for EOF
            f.seek(-1024, 2)
            tail = f.read()
            if b'%%EOF' not in tail:
                return False, "No EOF marker"

        return True, "PDF structure appears valid"
    except Exception as e:
        return False, str(e)

def repair_strategy_1_truncate_and_add_eof(filepath):
    """Strategy 1: Find last endobj and truncate there"""
    try:
        with open(filepath, 'rb') as f:
            content = f.read()

        # Find all endobj markers
        endobj_pattern = re.compile(b'endobj\\s*(\\d+\\s+\\d+\\s+obj)?')
        matches = list(endobj_pattern.finditer(content))

        if matches:
            # Get the last valid endobj
            last_match = matches[-1]
            end_pos = last_match.end()

            # Truncate and add EOF
            repaired = content[:end_pos]
            repaired += b'\n%%EOF\n'

            # Backup
            backup_path = os.path.join(BACKUP_DIR, os.path.basename(filepath))
            os.makedirs(BACKUP_DIR, exist_ok=True)
            shutil.copy2(filepath, backup_path)

            # Write repaired file
            with open(filepath, 'wb') as f:
                f.write(repaired)

            return True, f"Truncated at position {end_pos} and added EOF"
        else:
            return False, "No endobj markers found"

    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    print("PDF Repair Tool")
    print("=" * 80)

    corrupted_files = [
        "01_first_law_complexodynamics.pdf",
        "11_dilated_convolutions.pdf",
        "26_cs231n_lecture1.pdf"
    ]

    for filename in corrupted_files:
        filepath = os.path.join(PDF_DIR, filename)

        if not os.path.exists(filepath):
            print(f"\n[SKIP] {filename} - File not found")
            continue

        print(f"\n[REPAIRING] {filename}")
        print("-" * 80)

        # Verify first
        valid, msg = verify_pdf(filepath)
        print(f"Current status: {msg}")

        if valid:
            print("File is already valid, skipping repair")
            continue

        # Try repair strategies
        success = False
        strategy = 1

        # Strategy 1: Truncate at last endobj
        print(f"\nTrying Strategy {strategy}: Truncate at last endobj...")
        success, msg = repair_strategy_1_truncate_and_add_eof(filepath)
        if success:
            print(f"  Success: {msg}")
            # Verify repair
            valid, verify_msg = verify_pdf(filepath)
            print(f"  Verification: {verify_msg}")
            if valid:
                print(f"  [SUCCESS] {filename} repaired successfully!")
                continue

        # Strategy 2: Simple EOF addition
        strategy += 1
        print(f"\nTrying Strategy {strategy}: Simple EOF addition...")
        success, msg = repair_pdf_by_adding_eof(filepath)
        print(f"  Result: {msg}")

        # Verify final result
        valid, verify_msg = verify_pdf(filepath)
        print(f"\nFinal verification: {verify_msg}")

        if valid:
            print(f"[SUCCESS] {filename} repaired!")
        else:
            print(f"[FAILED] Could not repair {filename}")
            print(f"  Original file backed up to: {os.path.join(BACKUP_DIR, filename)}")

    print("\n" + "=" * 80)
    print("Repair process complete!")

if __name__ == "__main__":
    main()
