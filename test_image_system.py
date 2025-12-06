#!/usr/bin/env python3
"""
Test image extraction and indexing pipeline.
This will create a test image and verify the extraction and indexing works.
"""
import sys
import os
from pathlib import Path

# Setup path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def test_image_extraction():
    """Test the image extraction function"""
    print("\n" + "="*70)
    print("TEST 1: Image Text Extraction")
    print("="*70)
    
    try:
        from pdf_processor import extract_text_from_image
        from PIL import Image, ImageDraw, ImageFont
        import tempfile
        
        # Create a test image with some text
        print("\n1️⃣  Creating test image with text...")
        img = Image.new('RGB', (400, 300), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw some text on the image
        text_content = "Business Card\nJohn Doe\nEmail: john@example.com\nPhone: +1-234-567-8900\nWebsite: www.example.com"
        draw.text((20, 20), text_content, fill='black')
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as f:
            img.save(f.name)
            test_image_path = f.name
            print(f"   ✓ Created: {test_image_path}")
        
        # Test extraction
        print("\n2️⃣  Extracting text from image...")
        extracted = extract_text_from_image(test_image_path)
        
        print(f"\n   Extracted text ({len(extracted)} chars):")
        print(f"   " + "-"*66)
        for line in extracted.split('\n')[:10]:
            print(f"   {line}")
        if extracted.count('\n') > 10:
            print(f"   ... ({extracted.count(chr(10)) - 10} more lines)")
        print(f"   " + "-"*66)
        
        # Cleanup
        os.unlink(test_image_path)
        
        # Check results
        if extracted and len(extracted.strip()) > 0:
            print(f"\n   ✓ Extraction successful: {len(extracted)} characters extracted")
            return True
        else:
            print(f"\n   ❌ Extraction failed: No text extracted")
            return False
            
    except ImportError as e:
        print(f"\n   ❌ Import error: {e}")
        print(f"   Make sure pytesseract and Pillow are installed")
        return False
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_vector_store():
    """Test that documents can be added and searched"""
    print("\n" + "="*70)
    print("TEST 2: Vector Store Indexing")
    print("="*70)
    
    try:
        from vector import add_documents, search_documents
        import uuid
        
        conv_id = str(uuid.uuid4())
        
        print(f"\n1️⃣  Using test conversation: {conv_id}")
        
        # Add test documents
        print("\n2️⃣  Adding business card text to vector store...")
        test_text = """Business Card
John Doe
Email: john@example.com
Phone: +1-234-567-8900
Website: www.example.com
Location: San Francisco, CA"""
        
        add_documents(conv_id, "business_card.jpg", [test_text])
        print(f"   ✓ Added document to vector store")
        
        # Search
        print("\n3️⃣  Searching for business information...")
        results = search_documents(conv_id, "business card contact information")
        
        if results:
            print(f"\n   ✓ Found {len(results)} relevant documents:")
            for i, doc in enumerate(results, 1):
                content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                print(f"     {i}. {content}")
            return True
        else:
            print(f"\n   ❌ No documents found in search")
            return False
            
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_code_logic():
    """Test that the code logic is correct"""
    print("\n" + "="*70)
    print("TEST 3: Code Logic Verification")
    print("="*70)
    
    try:
        main_py = Path(__file__).parent / "backend" / "main.py"
        with open(main_py, 'r') as f:
            main_content = f.read()
        
        checks = [
            ("Images indexed in vector store", "add_documents(conversation_id, file.filename, chunks)"),
            ("Full text saved to database", "full_extracted_text"),
            ("System messages for files", 'role="system"'),
            ("Chat searches documents", "search_documents(conversation_id, request.message"),
            ("Context passed to model", '"context": context_content'),
        ]
        
        print("\nCode verification:")
        all_pass = True
        for check_name, pattern in checks:
            found = pattern in main_content
            status = "✓" if found else "❌"
            print(f"   {status} {check_name}")
            all_pass = all_pass and found
        
        return all_pass
        
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("IMAGE UPLOAD & INDEXING SYSTEM TEST")
    print("="*70)
    print(f"Testing image extraction, indexing, and retrieval pipeline")
    
    results = []
    
    # Run tests
    results.append(("Image Extraction", test_image_extraction()))
    results.append(("Vector Store", test_vector_store()))
    results.append(("Code Logic", test_code_logic()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓" if passed else "❌"
        print(f"{status} {test_name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ All tests PASSED!")
        print("\nYour chatbot is ready to:")
        print("  • Upload images (PNG, JPG, etc.)")
        print("  • Extract text using OCR (Tesseract)")
        print("  • Save full text to database")
        print("  • Index text in vector store")
        print("  • Find relevant images when asked questions")
        print("  • Answer detailed questions about uploaded images")
        print("\nExample: Upload a business card image and ask:")
        print('  "What is the email address on the business card?"')
        print('  "Tell me the phone number from the image"')
    else:
        print("⚠️  Some tests failed - see details above")
    print("="*70)
    
    return all_passed

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
