#!/bin/bash
#SBATCH --partition=compute
#SBATCH --nodes=1

# Path to sample list
SAMPLE_FILE="remaining_samples_to_process.txt"

# Read samples and submit jobs
while IFS= read -r SAMPLE; do
    # Create directories
    mkdir -p "./log/${SAMPLE}" "./error/${SAMPLE}" "./${SAMPLE}"

# Step 1: FastqToBam
JOB1=$(sbatch -J fgbio_ftb_${SAMPLE} --parsable \
  --cpus-per-task=2 \
  -o ./log/${SAMPLE}/fgbio_ftb.log \
  --error ./error/${SAMPLE}/fgbio_ftb.err \
  --time=0:40:00 \
  --mem-per-cpu=8000 \
  --wrap "fgbio -Xmx8g FastqToBam \
    --input ../shared_ap/hpv_capture_sp/${SAMPLE}/*R1_001.fastq.gz \
    ../shared_ap/hpv_capture_sp/${SAMPLE}/*R3_001.fastq.gz \
    ../shared_ap/hpv_capture_sp/${SAMPLE}/*R2_001.fastq.gz \
    --read-structures '+T' '+T' '9M' \
    --output ./${SAMPLE}/unmapped.bam \
    --sample ${SAMPLE} \
    --library l1 \
    --platform-unit HGGTLDRX5.1")

# Step 2: Alignment
JOB2=$(sbatch -J bwa_aln_${SAMPLE} --parsable \
  --dependency=afterok:${JOB1} \
  --cpus-per-task=2 \
  -o ./log/${SAMPLE}/bwa_aln.log \
  --error ./error/${SAMPLE}/bwa_aln.err \
  --time=1:10:00 \
  --mem-per-cpu=8000 \
  --wrap "samtools fastq -N -t ./${SAMPLE}/unmapped.bam | \
    bwa-mem2 mem -p -Y ./HPV_Ref/hpv.hpv33_HQ537690.1.hpv35_KX514416.1.short_names.fasta - | \
    samtools view -b > ./${SAMPLE}/aligned_raw.bam && \
    samtools sort ./${SAMPLE}/aligned_raw.bam -o ./${SAMPLE}/aligned_raw.sorted.bam && \
    samtools index ./${SAMPLE}/aligned_raw.sorted.bam")

# Step 3: ZipperBams
JOB3=$(sbatch -J zipper_${SAMPLE} --parsable \
  --dependency=afterok:${JOB2} \
  --cpus-per-task=2 \
  -o ./log/${SAMPLE}/zipper.log \
  --error ./error/${SAMPLE}/zipper.err \
  --time=0:40:00 \
  --mem-per-cpu=8000 \
  --wrap "fgbio ZipperBams \
    -i ./${SAMPLE}/aligned_raw.bam \
    -u ./${SAMPLE}/unmapped.bam \
    -r ./HPV_Ref/hpv.hpv33_HQ537690.1.hpv35_KX514416.1.short_names.fasta \
    -o ./${SAMPLE}/zipped.bam \
    --tags-to-reverse Consensus \
    --tags-to-revcomp Consensus && \
    samtools sort ./${SAMPLE}/zipped.bam -o ./${SAMPLE}/zipped.sorted.bam")

# Step 4: GroupReadsByUmi
JOB4=$(sbatch -J group_umi_${SAMPLE} --parsable \
  --dependency=afterok:${JOB3} \
  --cpus-per-task=2 \
  -o ./log/${SAMPLE}/group_umi.log \
  --error ./error/${SAMPLE}/group_umi.err \
  --time=0:50:00 \
  --mem-per-cpu=8000 \
  --wrap "fgbio GroupReadsByUmi \
    -i ./${SAMPLE}/zipped.sorted.bam \
    -o ./${SAMPLE}/grouped.bam \
    --strategy=adjacency \
    --edits=1 \
    --min-map-q=20 \
    --include-secondary=false \
    --include-supplementary=false")

# Step 5: CallMolecularConsensusReads
JOB5=$(sbatch -J consensus_${SAMPLE} --parsable \
  --dependency=afterok:${JOB4} \
  --cpus-per-task=2 \
  -o ./log/${SAMPLE}/consensus.log \
  --error ./error/${SAMPLE}/consensus.err \
  --time=1:00:00 \
  --mem-per-cpu=8000 \
  --wrap "fgbio CallMolecularConsensusReads \
    -i ./${SAMPLE}/grouped.bam \
    -o ./${SAMPLE}/consensus_unfiltered.bam \
    --min-input-base-quality 25 \
    --min-reads 3")

# Step 6: Map Consensus Reads
JOB6=$(sbatch -J map_cons_${SAMPLE} --parsable \
  --dependency=afterok:${JOB5} \
  --cpus-per-task=2 \
  -o ./log/${SAMPLE}/map_cons.log \
  --error ./error/${SAMPLE}/map_cons.err \
  --time=1:10:00 \
  --mem-per-cpu=8000 \
  --wrap "samtools fastq ./${SAMPLE}/consensus_unfiltered.bam | \
    bwa-mem2 mem -p -Y ./HPV_Ref/hpv.hpv33_HQ537690.1.hpv35_KX514416.1.short_names.fasta - | \
    fgbio -Xmx8g ZipperBams \
      -i /dev/stdin \
      -u ./${SAMPLE}/consensus_unfiltered.bam \
      -r ./HPV_Ref/hpv.hpv33_HQ537690.1.hpv35_KX514416.1.short_names.fasta \
      --tags-to-reverse Consensus \
      --tags-to-revcomp Consensus \
      -o ./${SAMPLE}/s1.cons.mapped.bam")

# Step 7: Sort/Index and Coverage
JOB7=$(sbatch -J sort_index_map_cons_${SAMPLE} --parsable \
  --dependency=afterok:${JOB6} \
  --cpus-per-task=2 \
  -o ./log/${SAMPLE}/sort_index_map_cons.log \
  --error ./error/${SAMPLE}/sort_index_map_cons.err \
  --time=0:30:00 \
  --mem-per-cpu=8000 \
  --wrap "samtools sort ./${SAMPLE}/s1.cons.mapped.bam -o ./${SAMPLE}/s1.cons.mapped.sorted.bam && \
    samtools index ./${SAMPLE}/s1.cons.mapped.sorted.bam && \
    samtools depth -aa ./${SAMPLE}/s1.cons.mapped.sorted.bam > ./${SAMPLE}/hpv_consensus_coverage.tsv")

# Step 8: GATK Variant Calling
#JOB8=$(sbatch -J gatk_${SAMPLE} --parsable \
#  --dependency=afterok:${JOB7} \
#  --cpus-per-task=2 \
#  -o ./log/${SAMPLE}/gatk.log \
#  --error ./error/${SAMPLE}/gatk.err \
#  --time=1:00:00 \
#  --mem-per-cpu=8000 \
#  --wrap "gatk HaplotypeCaller \
#    -R ./HPV_Ref/hpv.hpv33_HQ537690.1.hpv35_KX514416.1.short_names.fasta \
#    -I ./${SAMPLE}/s1.cons.mapped.sorted.bam \
#    --ploidy 1 \
#    --emit-ref-confidence BP_RESOLUTION \
#    -O ./${SAMPLE}/hpv_gatk_all_sites.vcf.gz \
#    --create-output-variant-index")

# Step 9: FreeBayes Variant Calling
#sbatch -J freebayes_${SAMPLE} \
#  --dependency=afterok:${JOB7} \
#  --cpus-per-task=2 \
#  -o ./log/${SAMPLE}/freebayes.log \
#  --error ./error/${SAMPLE}/freebayes.err \
#  --time=0:50:00 \
#  --mem-per-cpu=8000 \
#  --wrap "freebayes \
#    -f ./HPV_Ref/hpv.hpv33_HQ537690.1.hpv35_KX514416.1.short_names.fasta \
#    --ploidy 1 \
#    --pooled-continuous \
#    --min-base-quality 20 \
#    ./${SAMPLE}/s1.cons.mapped.sorted.bam > ./${SAMPLE}/hpv_freebayes.vcf"
done < "${SAMPLE_FILE}"
