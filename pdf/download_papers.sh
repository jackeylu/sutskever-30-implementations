#!/bin/bash
# Download 30 papers from Ilya Sutskever's reading list

PDF_DIR="D:/codes/learn/pdf/pdf"
cd "$PDF_DIR" || exit 1

echo "Starting download of papers to $PDF_DIR"
echo "============================================================"

# Function to download a file
download_file() {
    local filename="$1"
    local url="$2"

    if [ -z "$url" ]; then
        echo "Skipping $filename (no URL provided)"
        return
    fi

    echo "Downloading $filename..."
    if curl -L -o "$filename" --max-time 60 --retry 3 "$url" 2>/dev/null; then
        if [ -f "$filename" ] && [ "$?" -eq 0 ]; then
            size=$(wc -c < "$filename" 2>/dev/null || echo "0")
            if [ "$size" -gt 1000 ]; then
                echo "OK: Downloaded $filename ($(numfmt --to=iec-i --suffix=B $size 2>/dev/null || echo "$size bytes"))"
            else
                echo "SKIP: $filename (file too small, possibly error page)"
                rm -f "$filename"
            fi
        fi
    else
        echo "FAIL: Failed to download $filename"
    fi
    echo ""
}

# Download all papers
download_file "01_first_law_complexodynamics.pdf" "https://www.scottaaronson.com/papers/coffee2.pdf"
download_file "02_unreasonable_effectiveness_rnns.pdf" "https://web.stanford.edu/class/cs379c/archive/2018/class_messages_listing/content/Artificial_Neural_Network_Technology_Tutorials/KarparthyUNREASONABLY-EFFECTIVE-RNN-15.pdf"
# Paper 3: Understanding LSTM (blog post, no PDF)
download_file "04_rnn_regularization.pdf" "https://arxiv.org/pdf/1409.2329.pdf"
download_file "05_neural_networks_simple.pdf" "https://www.cs.toronto.edu/~fritz/absps/colt93.pdf"
download_file "06_pointer_networks.pdf" "https://arxiv.org/pdf/1506.03134.pdf"
download_file "07_alexnet.pdf" "https://proceedings.neurips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf"
download_file "08_order_matters_seq2seq.pdf" "https://arxiv.org/pdf/1511.06391.pdf"
download_file "09_gpipe.pdf" "https://arxiv.org/pdf/1811.06965.pdf"
download_file "10_resnet.pdf" "https://arxiv.org/pdf/1512.03385.pdf"
download_file "11_dilated_convolutions.pdf" "http://vladlen.info/papers/dilated-convolutions.pdf"
download_file "12_neural_message_passing.pdf" "https://arxiv.org/pdf/1704.01212.pdf"
download_file "13_attention_is_all_you_need.pdf" "https://arxiv.org/pdf/1706.03762.pdf"
download_file "14_bahdanau_attention.pdf" "https://arxiv.org/pdf/1409.0473.pdf"
download_file "15_identity_mappings_resnet.pdf" "https://arxiv.org/pdf/1603.05027.pdf"
download_file "16_relational_reasoning.pdf" "https://arxiv.org/pdf/1706.01427.pdf"
download_file "17_variational_lossy_autoencoder.pdf" "https://arxiv.org/pdf/1611.02731.pdf"
download_file "18_relational_rnn.pdf" "https://arxiv.org/pdf/1806.01822.pdf"
download_file "19_coffee_automaton.pdf" "https://www.scottaaronson.com/papers/coffee2.pdf"
download_file "20_neural_turing_machines.pdf" "https://arxiv.org/pdf/1410.5401.pdf"
download_file "21_deep_speech_2.pdf" "https://arxiv.org/pdf/1512.02595.pdf"
download_file "22_scaling_laws.pdf" "https://arxiv.org/pdf/2001.08361.pdf"
download_file "23_mdl_principle.pdf" "https://homepages.cwi.nl/~paulv/course-kc/mdlintro.pdf"
download_file "24_machine_super_intelligence.pdf" "http://www.vetta.org/documents/Machine_Super_Intelligence.pdf"
download_file "25_kolmogorov_complexity.pdf" "https://arxiv.org/pdf/1504.04955.pdf"
download_file "26_cs231n_lecture1.pdf" "https://cs231n.stanford.edu/2021/slides/2021/lecture_1.pdf"
download_file "27_multi_token_prediction.pdf" "https://arxiv.org/pdf/2404.19737.pdf"
download_file "28_dense_passage_retrieval.pdf" "https://arxiv.org/pdf/2004.04906.pdf"
download_file "29_retrieval_augmented_generation.pdf" "https://arxiv.org/pdf/2005.11401.pdf"
download_file "30_lost_in_the_middle.pdf" "https://arxiv.org/pdf/2307.03172.pdf"

echo "============================================================"
echo "Download complete!"
echo ""
echo "Files downloaded:"
ls -lh *.pdf 2>/dev/null | awk '{print $9, "-", $5}'
