num_experiments=100

experiments1000=(experiment1 experiment2 experiment3 experiment4 experiment5 experiment6 experiment7 experiment8 experiment9 experiment10 experiment11 experiment12 experiment13 experiment14 experiment15 experiment16 experiment17 experiment18)

experiments20000=(experiment1 experiment2 experiment3 experiment4 experiment5 experiment6 experiment7 experiment8 experiment9 experiment10 experiment11 experiment12 experiment13 experiment14 experiment15 experiment16 experiment17 experiment18)

experiments100000=(experiment16)

allmodselalgos=DoublingDataDriven,EstimatingDataDriven,Corral,BalancingClassic,UCB,Greedy,EXP3,CorralLow,CorralHigh,EXP3Low,EXP3High,EXP3LowLR,EXP3HighLR,DoublingDataDrivenMedium,EstimatingDataDrivenMedium,DoublingDataDrivenHigh,EstimatingDataDrivenHigh
corralvariants=Corral,CorralLow,CorralHigh
exp3variants=EXP3,EXP3Low,EXP3High,EXP3LowLR,EXP3HighLR
doublingvariants=DoublingDataDriven,DoublingDataDrivenMedium,DoublingDataDrivenHigh
estimatedvariants=EstimatingDataDriven,EstimatingDataDrivenMedium,EstimatingDataDrivenHigh

#bash commit.sh;


# for experiment in "${experiments1000[@]}"; 
# do
# 	python experiments_synthetic.py 1000 $experiment $num_experiments $allmodselalgos True;
# 	python experiments_synthetic.py 1000 $experiment $num_experiments $corralvariants True;	
# 	python experiments_synthetic.py 1000 $experiment $num_experiments $exp3variants True;
# 	python experiments_synthetic.py 1000 $experiment $num_experiments $doublingvariants True;
# 	python experiments_synthetic.py 1000 $experiment $num_experiments $estimatedvariants True;

# done

# bash commit.sh;



for experiment in "${experiments20000[@]}"; 
do
	python experiments_synthetic.py 20000 $experiment $num_experiments $allmodselalgos True;
	python experiments_synthetic.py 20000 $experiment $num_experiments $corralvariants True;	
	python experiments_synthetic.py 20000 $experiment $num_experiments $exp3variants True;
	python experiments_synthetic.py 20000 $experiment $num_experiments $doublingvariants True;
	python experiments_synthetic.py 20000 $experiment $num_experiments $estimatedvariants True;
done

bash commit.sh;


for experiment in "${experiments100000[@]}"; 
do
	python experiments_synthetic.py 100000 $experiment $num_experiments $allmodselalgos True;
	python experiments_synthetic.py 100000 $experiment $num_experiments $corralvariants True;	
	python experiments_synthetic.py 100000 $experiment $num_experiments $exp3variants True;
	python experiments_synthetic.py 100000 $experiment $num_experiments $doublingvariants True;
	python experiments_synthetic.py 100000 $experiment $num_experiments $estimatedvariants True;
done

bash commit.sh;
