num_experiments=100

experiments1000=(experiment1 experiment2 experiment3 experiment4 experiment5 experiment6 experiment7 experiment8 experiment9 experiment10 experiment11 experiment12 experiment13 experiment14 experiment15 experiment16 experiment17 experiment18)

experiments20000=(experiment1 experiment2 experiment3 experiment4 experiment5 experiment6 experiment7 experiment8 experiment9 experiment10 experiment11 experiment12 experiment13 experiment14 experiment15 experiment16 experiment17 experiment18)

experiments100000=(experiment16)


modselalgospaper=DoublingDataDriven,EstimatingDataDriven,Corral,BalancingClassic-g,UCB,Greedy,EXP3
allmodselalgos=DoublingDataDriven,EstimatingDataDriven,Corral,BalancingClassic-g,UCB,Greedy,EXP3,CorralLow,CorralHigh,EXP3Low,EXP3High,EXP3LowLR,EXP3HighLR,DoublingDataDrivenMedium,EstimatingDataDrivenMedium,DoublingDataDrivenHigh,EstimatingDataDrivenHigh,CorralSuperHigh,DoublingDataDrivenSuperHigh,EstimatingDataDrivenSuperHigh
allmodselalgos_shared=DoublingDataDriven-s,EstimatingDataDriven-s,Corral-s,BalancingClassic-gs,UCB-s,Greedy-s,EXP3-s,CorralLow-s,CorralHigh-s,EXP3Low-s,EXP3High-s,EXP3LowLR-s,EXP3HighLR-s,DoublingDataDrivenMedium-s,EstimatingDataDrivenMedium-s,DoublingDataDrivenHigh-s,EstimatingDataDrivenHigh-s,CorralSuperHigh-s,DoublingDataDrivenSuperHigh-s,EstimatingDataDrivenSuperHigh-s
corralvariants=Corral,CorralLow,CorralHigh
corralvariants_shared=Corral-s,CorralLow-s,CorralHigh-s
exp3variants=EXP3,EXP3Low,EXP3High,EXP3LowLR,EXP3HighLR
exp3variants_shared=EXP3-s,EXP3Low-s,EXP3High-s,EXP3LowLR-s,EXP3HighLR-s
doublingvariants=DoublingDataDriven,DoublingDataDrivenMedium,DoublingDataDrivenHigh
doublingvariants_shared=DoublingDataDriven-s,DoublingDataDrivenMedium-s,DoublingDataDrivenHigh-s
estimatedvariants=EstimatingDataDriven,EstimatingDataDrivenMedium,EstimatingDataDrivenHigh
estimatedvariants_shared=EstimatingDataDriven-s,EstimatingDataDrivenMedium-s,EstimatingDataDrivenHigh-s
papervariants=DoublingDataDriven,EstimatingDataDriven,Corral,BalancingClassic-g,UCB,Greedy,EXP3
papervariants_shared=DoublingDataDriven-s,EstimatingDataDriven-s,Corral-s,BalancingClassic-gs,UCB-s,Greedy-s,EXP3-s
superhigh=CorralSuperHigh,DoublingDataDrivenSuperHigh,EstimatingDataDrivenSuperHigh
superhigh_shared=CorralSuperHigh-s,DoublingDataDrivenSuperHigh-s,EstimatingDataDrivenSuperHigh-s

#bash commit.sh;
stochsamplingvariants=DoublingDataDrivenStoch,EstimatingDataDrivenStoch
stochsamplingvariants_shared=DoublingDataDrivenStoch-s,EstimatingDataDrivenStoch-s

for experiment in "${experiments1000[@]}"; 
do
	python experiments_synthetic.py 1000 $experiment $num_experiments $modselalgospaper True;
done

bash commit.sh;


for experiment in "${experiments20000[@]}"; 
do
	python experiments_synthetic.py 20000 $experiment $num_experiments $modselalgospaper True;
done

bash commit.sh;


for experiment in "${experiments100000[@]}"; 
do
	python experiments_synthetic.py 100000 $experiment $num_experiments $modselalgospaper True;
done

bash commit.sh;
