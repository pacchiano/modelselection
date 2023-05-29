num_experiments=100

experiments1000=experiment1,experiment2,experiment3,experiment4,experiment5,experiment6,experiment7,experiment8,experiment9,experiment10,experiment11,experiment12,experiment13,experiment14,experiment15,experiment16,experiment17,experiment18

experiments20000=experiment1,experiment2,experiment3,experiment4,experiment5,experiment6,experiment7,experiment8,experiment9,experiment10,experiment11,experiment12,experiment13,experiment14,experiment15,experiment16,experiment17,experiment18

experiments100000=experiment16

allmodselalgos=DoublingDataDriven,EstimatingDataDriven,Corral,BalancingClassic,UCB,Greedy,EXP3,CorralLow,CorralHigh,EXP3Low,EXP3High,EXP3LowLR,EXP3HighLR,DoublingDataDrivenMedium,EstimatingDataDrivenMedium,DoublingDataDrivenHigh,EstimatingDataDrivenHigh,CorralSuperHigh,DoublingDataDrivenSuperHigh,EstimatingDataDrivenSuperHigh
corralvariants=Corral,CorralLow,CorralHigh,CorralSuperHigh
exp3variants=EXP3,EXP3Low,EXP3High,EXP3LowLR,EXP3HighLR
doublingvariants=DoublingDataDriven,DoublingDataDrivenMedium,DoublingDataDrivenHigh,DoublingDataDrivenSuperHigh
estimatedvariants=EstimatingDataDriven,EstimatingDataDrivenMedium,EstimatingDataDrivenHigh,EstimatingDataDrivenSuperHigh
papervariants=DoublingDataDriven,EstimatingDataDriven,Corral,BalancingClassic,UCB,Greedy,EXP3
#bash commit.sh;


# python table_write.py 1000 $experiments1000 $num_experiments $allmodselalgos $allmodselalgos True;
# python table_write.py 1000 $experiments1000 $num_experiments $allmodselalgos $corralvariants True;	



python table_write.py 20000 $experiments20000 $num_experiments $allmodselalgos $allmodselalgos True;
python table_write.py 20000 $experiments20000 $num_experiments $allmodselalgos $corralvariants True;	
python table_write.py 20000 $experiments20000 $num_experiments $allmodselalgos $papervariants True;	


python table_write.py 100000 $experiments100000 $num_experiments $allmodselalgos $allmodselalgos True;
python table_write.py 100000 $experiments100000 $num_experiments $allmodselalgos $corralvariants True;	
python table_write.py 100000 $experiments100000 $num_experiments $allmodselalgos $papervariants True;	

