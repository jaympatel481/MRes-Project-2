#!/bin/bash

# Check if vaf_list.txt exists
if [ ! -f "samples.txt" ]; then
    echo "Error: samples.txt not found!"
    exit 1
fi

# Read each subdirectory name from the file
while IFS= read -r SAMPLE; do
    # Skip empty lines
    if [ -z "$SAMPLE" ]; then
        continue
    fi # CORRECTED: Changed SFFI to fi
    
    echo "Processing directory: $SAMPLE"
    
    # Construct the full path to the input VCF file
    INPUT_VCF="${SAMPLE}/hpv_somatic_snvs.vcf.gz"
    bgzip -c ./${SAMPLE}/hpv_somatic_snvs.vcf > "${INPUT_VCF}"

    # Construct the full path for the output annotated VCF file
    OUTPUT_VCF="${SAMPLE}/hpv_somatic_sb_posb_mina_rep_fil_ann.vcf.gz"
    
    # Check if the input VCF file exists before running snpEff
    if [ -f "$INPUT_VCF" ]; then
        echo "Running snpEff for $SAMPLE..."
        
        snpEff eff HPV16 "${INPUT_VCF}" > "${OUTPUT_VCF}"
        if [ $? -eq 0 ]; then
            echo "Successfully annotated: ${OUTPUT_VCF}"
        else
            echo "Error running snpEff for $SAMPLE. Check previous messages."
        fi
    else
        echo "Warning: Input VCF file not found for $SAMPLE: ${INPUT_VCF}"
    fi

done < "samples.txt"

echo "Script finished."
