#!/usr/bin/env python3
"""
Check PDF files for corruption and incompleteness
"""

import os
import struct

PDF_DIR = "D:/codes/learn/pdf/pdf"

def check_pdf_header(filepath):
    """Check if file has valid PDF header"""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                return False, "Invalid PDF header"
            return True, "Valid header"
    except Exception as e:
        return False, str(e)

def check_pdf_eof(filepath):
    """Check if file has valid PDF end-of-file marker"""
    try:
        with open(filepath, 'rb') as f:
            # Read last 1024 bytes
            f.seek(-1024, 2)  # Seek to 1024 bytes from end
            tail = f.read()
            if b'%%EOF' in tail:
                return True, "Has EOF marker"
            else:
                return False, "Missing EOF marker (incomplete download)"
    except Exception as e:
        return False, str(e)

def get_file_size_mb(filepath):
    """Get file size in MB"""
    try:
        size = os.path.getsize(filepath)
        return size / (1024 * 1024)
    except:
        return 0

def analyze_pdf(filepath):
    """Analyze PDF file for issues"""
    filename = os.path.basename(filepath)
    size_mb = get_file_size_mb(filepath)

    header_valid, header_msg = check_pdf_header(filepath)
    eof_valid, eof_msg = check_pdf_eof(filepath)

    # Determine if file is likely corrupted
    issues = []
    if not header_valid:
        issues.append(f"Header: {header_msg}")
    if not eof_valid:
        issues.append(f"EOF: {eof_msg}")

    # Check if size is suspiciously small
    if size_mb < 0.01:  # Less than 10KB
        issues.append(f"File too small ({size_mb:.3f} MB)")

    return {
        'filename': filename,
        'size_mb': size_mb,
        'header_valid': header_valid,
        'eof_valid': eof_valid,
        'issues': issues,
        'status': 'OK' if len(issues) == 0 else 'CORRUPTED'
    }

def main():
    print("Checking PDF files for corruption...")
    print("=" * 80)

    pdf_files = [f for f in os.listdir(PDF_DIR) if f.endswith('.pdf')]

    results = []
    for filename in sorted(pdf_files):
        filepath = os.path.join(PDF_DIR, filename)
        result = analyze_pdf(filepath)
        results.append(result)

        # Print status
        status_symbol = "[OK]" if result['status'] == 'OK' else "[BAD]"
        print(f"{status_symbol} {result['filename']}")
        print(f"  Size: {result['size_mb']:.2f} MB")
        if result['issues']:
            print(f"  Issues: {', '.join(result['issues'])}")
        print()

    # Summary
    print("=" * 80)
    ok_count = sum(1 for r in results if r['status'] == 'OK')
    corrupted_count = sum(1 for r in results if r['status'] == 'CORRUPTED')

    print(f"Total PDFs: {len(results)}")
    print(f"OK: {ok_count}")
    print(f"Corrupted/Incomplete: {corrupted_count}")

    if corrupted_count > 0:
        print("\nCorrupted files:")
        for r in results:
            if r['status'] == 'CORRUPTED':
                print(f"  - {r['filename']} ({r['size_mb']:.2f} MB)")
                print(f"    Issues: {', '.join(r['issues'])}")

if __name__ == "__main__":
    main()
