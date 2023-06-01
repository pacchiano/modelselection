num_experiments=100

experiments1000=(experiment1 experiment2 experiment3 experiment4 experiment5 experiment6 experiment7 experiment8 experiment9 experiment10 experiment11 experiment12 experiment13 experiment14 experiment15 experiment16 experiment17 experiment18)

experiments20000=(experiment1 experiment2 experiment3 experiment4 experiment5 experiment6 experiment7 experiment8 experiment9 experiment10 experiment11 experiment12 experiment13 experiment14 experiment15 experiment16 experiment17 experiment18)

experiments100000=(experiment16)

allmodselalgos=DoublingDataDriven,EstimatingDataDriven,Corral,BalancingClassic,UCB,Greedy,EXP3,CorralLow,CorralHigh,EXP3Low,EXP3High,EXP3LowLR,EXP3HighLR,DoublingDataDrivenMedium,EstimatingDataDrivenMedium,DoublingDataDrivenHigh,EstimatingDataDrivenHigh,CorralSuperHigh,DoublingDataDrivenSuperHigh,EstimatingDataDrivenSuperHigh
corralvariants=Corral,CorralLow,CorralHigh
exp3variants=EXP3,EXP3Low,EXP3High,EXP3LowLR,EXP3HighLR
doublingvariants=DoublingDataDriven,DoublingDataDrivenMedium,DoublingDataDrivenHigh
estimatedvariants=EstimatingDataDriven,EstimatingDataDrivenMedium,EstimatingDataDrivenHigh
papervariants=DoublingDataDriven,EstimatingDataDriven,Corral,BalancingClassic,UCB,Greedy,EXP3
superhigh=CorralSuperHigh,DoublingDataDrivenSuperHigh,EstimatingDataDrivenSuperHigh
#bash commit.sh;
stochsamplingvariants=DoublingDataDrivenStoch,EstimatingDataDrivenStoch

for experiment in "${experiments1000[@]}"; 
do
	python experiments_synthetic.py 1000 $experiment $num_experiments $allmodselalgos True True;
	python experiments_synthetic.py 1000 $experiment $num_experiments $corralvariants True True;	
	python experiments_synthetic.py 1000 $experiment $num_experiments $exp3variants True True;
	python experiments_synthetic.py 1000 $experiment $num_experiments $doublingvariants True True;
	python experiments_synthetic.py 1000 $experiment $num_experiments $estimatedvariants True True;
	python experiments_synthetic.py 1000 $experiment $num_experiments $papervariants True True;
	python experiments_synthetic.py 1000 $experiment $num_experiments $stochsamplingvariants True True;
done

bash commit.sh;



for experiment in "${experiments20000[@]}"; 
do
	python experiments_synthetic.py 20000 $experiment $num_experiments $allmodselalgos True False;
	python experiments_synthetic.py 20000 $experiment $num_experiments $corralvariants True False;	
	python experiments_synthetic.py 20000 $experiment $num_experiments $exp3variants True False;
	python experiments_synthetic.py 20000 $experiment $num_experiments $doublingvariants True False;
	python experiments_synthetic.py 20000 $experiment $num_experiments $estimatedvariants True False;
	python experiments_synthetic.py 20000 $experiment $num_experiments $papervariants True False;
	python experiments_synthetic.py 20000 $experiment $num_experiments $stochsamplingvariants True False;

done

bash commit.sh;


for experiment in "${experiments100000[@]}"; 
do
	python experiments_synthetic.py 100000 $experiment $num_experiments $allmodselalgos True False;
	python experiments_synthetic.py 100000 $experiment $num_experiments $corralvariants True False;	
	python experiments_synthetic.py 100000 $experiment $num_experiments $exp3variants True False;
	python experiments_synthetic.py 100000 $experiment $num_experiments $doublingvariants True False;
	python experiments_synthetic.py 100000 $experiment $num_experiments $estimatedvariants True False;
	python experiments_synthetic.py 100000 $experiment $num_experiments $papervariants True False;
	python experiments_synthetic.py 100000 $experiment $num_experiments $stochsamplingvariants True False;

done

bash commit.sh;
