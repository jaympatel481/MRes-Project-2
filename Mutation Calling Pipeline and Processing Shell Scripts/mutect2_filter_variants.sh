#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --output=%x-%j.out

SAMPLE_FILE="samples.txt"
REF_FILE="./HPV_Ref/hpv.hpv33_HQ537690.1.hpv35_KX514416.1.short_names.fasta"

set -eo pipefail  # Exit on error, preserve error messages

while IFS= read -r SAMPLE; do
    echo "Processing sample: ${SAMPLE}"

    # Define file paths with proper quoting
    INPUT_VCF="./${SAMPLE}/hpv_somatic.vcf.gz"
    SPLIT_VCF="./${SAMPLE}/hpv_somatic_split_before_filter.vcf"
    SPLIT_VCF_GZ="./${SAMPLE}/hpv_somatic_split_before_filter.vcf.gz"
    REPEAT_FIL_VCF="./${SAMPLE}/hpv_somatic_repeat_fil.vcf"
    REPEAT_FIL_VCF_GZ="./${SAMPLE}/hpv_somatic_repeat_fil.vcf.gz"
    FILTERED_VCF="./${SAMPLE}/hpv_somatic_filtered.vcf"
    FILTERED_VCF_GZ="./${SAMPLE}/hpv_somatic_filtered.vcf.gz"

    # Step 1: Normalize and split multi-allelic variants
    echo "Step 1: Splitting variants..."
    bcftools norm -m-any --check-ref -w -f "$REF_FILE" "$INPUT_VCF" -o "$SPLIT_VCF"
    bgzip "$SPLIT_VCF"
    tabix -p vcf "$SPLIT_VCF_GZ"

    # Step 2: Filter out repeat positions
    echo "Step 2: Excluding repeat positions..."
    bcftools view -o "$REPEAT_FIL_VCF" -T ^repeat_positions.bed "$SPLIT_VCF_GZ"
    bgzip "$REPEAT_FIL_VCF"
    tabix -p vcf "$REPEAT_FIL_VCF_GZ"

    # Step 3: Apply custom filters
    echo "Step 3: Applying quality filters..."
    bcftools filter -o "$FILTERED_VCF" "$REPEAT_FIL_VCF_GZ" \
        -r HPV16 \
        -e "MPOS <= 15 || FORMAT/DP[0] < 10 || FORMAT/AD[0:1] < 3 || \
           (FORMAT/SB[0:2]/(FORMAT/SB[0:3]+FORMAT/SB[0:2])) < 0.3 || \
	   (FORMAT/SB[0:2]/(FORMAT/SB[0:3]+FORMAT/SB[0:2])) > 0.7"

    bgzip "$FILTERED_VCF"
    tabix -p vcf "$FILTERED_VCF_GZ"

done < "$SAMPLE_FILE"

