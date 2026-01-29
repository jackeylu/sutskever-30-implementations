#!/usr/bin/env python3
"""
Download 30 papers from Ilya Sutskever's reading list
"""

import os
import urllib.request
import urllib.error

# Create pdf directory
PDF_DIR = "D:/codes/learn/pdf/pdf"
os.makedirs(PDF_DIR, exist_ok=True)

# List of papers with their download URLs
papers = [
    # Paper 1: The First Law of Complexodynamics (blog post, downloading related paper)
    ("01_first_law_complexodynamics.pdf", "https://www.scottaaronson.com/papers/coffee2.pdf"),

    # Paper 2: The Unreasonable Effectiveness of RNNs
    ("02_unreasonable_effectiveness_rnns.pdf", "https://web.stanford.edu/class/cs379c/archive/2018/class_messages_listing/content/Artificial_Neural_Network_Technology_Tutorials/KarparthyUNREASONABLY-EFFECTIVE-RNN-15.pdf"),

    # Paper 3: Understanding LSTM Networks (blog post - skip, no official PDF)
    # ("03_understanding_lstm.pdf", ""),

    # Paper 4: Recurrent Neural Network Regularization
    ("04_rnn_regularization.pdf", "https://arxiv.org/pdf/1409.2329.pdf"),

    # Paper 5: Keeping Neural Networks Simple
    ("05_neural_networks_simple.pdf", "https://www.cs.toronto.edu/~fritz/absps/colt93.pdf"),

    # Paper 6: Pointer Networks
    ("06_pointer_networks.pdf", "https://arxiv.org/pdf/1506.03134.pdf"),

    # Paper 7: AlexNet
    ("07_alexnet.pdf", "https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf"),

    # Paper 8: Order Matters: Sequence to Sequence for Sets
    ("08_order_matters_seq2seq.pdf", "https://arxiv.org/pdf/1511.06391.pdf"),

    # Paper 9: GPipe
    ("09_gpipe.pdf", "https://arxiv.org/pdf/1811.06965.pdf"),

    # Paper 10: Deep Residual Learning (ResNet)
    ("10_resnet.pdf", "https://arxiv.org/pdf/1512.03385.pdf"),

    # Paper 11: Multi-Scale Context Aggregation (Dilated Convolutions)
    ("11_dilated_convolutions.pdf", "http://vladlen.info/papers/dilated-convolutions.pdf"),

    # Paper 12: Neural Message Passing for Quantum Chemistry
    ("12_neural_message_passing.pdf", "https://arxiv.org/pdf/1704.01212.pdf"),

    # Paper 13: Attention Is All You Need
    ("13_attention_is_all_you_need.pdf", "https://arxiv.org/pdf/1706.03762.pdf"),

    # Paper 14: Neural Machine Translation (Bahdanau Attention)
    ("14_bahdanau_attention.pdf", "https://arxiv.org/pdf/1409.0473.pdf"),

    # Paper 15: Identity Mappings in Deep Residual Networks
    ("15_identity_mappings_resnet.pdf", "https://arxiv.org/pdf/1603.05027.pdf"),

    # Paper 16: Simple Neural Network for Relational Reasoning
    ("16_relational_reasoning.pdf", "https://arxiv.org/pdf/1706.01427.pdf"),

    # Paper 17: Variational Lossy Autoencoder
    ("17_variational_lossy_autoencoder.pdf", "https://arxiv.org/pdf/1611.02731.pdf"),

    # Paper 18: Relational Recurrent Neural Networks
    ("18_relational_rnn.pdf", "https://arxiv.org/pdf/1806.01822.pdf"),

    # Paper 19: The Coffee Automaton
    ("19_coffee_automaton.pdf", "https://www.scottaaronson.com/papers/coffee2.pdf"),

    # Paper 20: Neural Turing Machines
    ("20_neural_turing_machines.pdf", "https://arxiv.org/pdf/1410.5401.pdf"),

    # Paper 21: Deep Speech 2
    ("21_deep_speech_2.pdf", "https://arxiv.org/pdf/1512.02595.pdf"),

    # Paper 22: Scaling Laws for Neural Language Models
    ("22_scaling_laws.pdf", "https://arxiv.org/pdf/2001.08361.pdf"),

    # Paper 23: Minimum Description Length Principle
    ("23_mdl_principle.pdf", "https://homepages.cwi.nl/~paulv/course-kc/mdlintro.pdf"),

    # Paper 24: Machine Super Intelligence
    ("24_machine_super_intelligence.pdf", "http://www.vetta.org/documents/Machine_Super_Intelligence.pdf"),

    # Paper 25: Kolmogorov Complexity (book - using intro paper)
    ("25_kolmogorov_complexity.pdf", "https://arxiv.org/pdf/1504.04955.pdf"),

    # Paper 26: CS231n (course - downloading lecture slides)
    ("26_cs231n_lecture1.pdf", "https://cs231n.stanford.edu/2021/slides/2021/lecture_1.pdf"),

    # Paper 27: Multi-token Prediction
    ("27_multi_token_prediction.pdf", "https://arxiv.org/pdf/2404.19737.pdf"),

    # Paper 28: Dense Passage Retrieval
    ("28_dense_passage_retrieval.pdf", "https://arxiv.org/pdf/2004.04906.pdf"),

    # Paper 29: Retrieval-Augmented Generation
    ("29_retrieval_augmented_generation.pdf", "https://arxiv.org/pdf/2005.11401.pdf"),

    # Paper 30: Lost in the Middle
    ("30_lost_in_the_middle.pdf", "https://arxiv.org/pdf/2307.03172.pdf"),
]

def download_file(url, filename, timeout=30):
    """Download a file from URL to filename with timeout"""
    try:
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, os.path.join(PDF_DIR, filename))
        print(f"✓ Successfully downloaded {filename}")
        return True
    except urllib.error.HTTPError as e:
        print(f"✗ HTTP Error for {filename}: {e.code}")
        return False
    except urllib.error.URLError as e:
        print(f"✗ URL Error for {filename}: {e.reason}")
        return False
    except Exception as e:
        print(f"✗ Error downloading {filename}: {str(e)}")
        return False

def main():
    print(f"Starting download of {len(papers)} papers to {PDF_DIR}")
    print("=" * 60)

    successful = 0
    failed = 0

    for filename, url in papers:
        if url:  # Only download if URL is provided
            if download_file(url, filename):
                successful += 1
            else:
                failed += 1
        else:
            print(f"⊘ Skipping {filename} (no URL provided)")

        print()  # Empty line between downloads

    print("=" * 60)
    print(f"Download complete!")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"⊘ Skipped: {len(papers) - successful - failed}")

if __name__ == "__main__":
    main()
