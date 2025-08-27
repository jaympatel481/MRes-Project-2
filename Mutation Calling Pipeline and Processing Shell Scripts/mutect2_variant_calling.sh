#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=1

SAMPLE_FILE="remaining_samples_to_process.txt"

# Read samples and submit jobs
while IFS= read -r SAMPLE; do

    sbatch -J "mutect2_${SAMPLE}" \
        --cpus-per-task 2 \
        -o "./log/${SAMPLE}/mutect2.log" \
        --error "./error/${SAMPLE}/mutect2.err" \
        --time 0:30:00 \
        --mem-per-cpu 8000 \
        --wrap "gatk Mutect2 \
            -R ./HPV_Ref/hpv.hpv33_HQ537690.1.hpv35_KX514416.1.short_names.fasta \
            -I ./${SAMPLE}/s1.cons.mapped.sorted.bam \
            --panel-of-normals ./PON/hpv_pon.vcf.gz \
            --germline-resource ./PON/hpv_pon.vcf.gz \
            --af-of-alleles-not-in-resource 0.000001 \
            --disable-read-filter MateOnSameContigOrNoMappedMateReadFilter \
            -O ./${SAMPLE}/hpv_somatic.vcf.gz"

done < "${SAMPLE_FILE}"

