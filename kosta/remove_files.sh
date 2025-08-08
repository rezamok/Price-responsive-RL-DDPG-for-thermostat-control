counter=50
while [ $counter -le 9000 ]
do
	echo $counter
#	rm ./checkpoints/mae-clipped-no-zero-uk/alg1/episode_${counter}.pt
#    rm ./checkpoints/mae-clipped-no-zero-uk/alg1/plots/eval_comf_viol_${counter}.png
#    rm ./checkpoints/mae-clipped-no-zero-uk/alg1/plots/reward_${counter}.png
    rm ./checkpoints/mae-clipped-no-zero-uk/alg1/plots/eval_rewad_${counter}.png
    rm ./checkpoints/mae-clipped-no-zero-uk/alg1/plots/net2loss_${counter}.png
#    rm ./checkpoints/mae-clipped-no-zero-uk/alg1/plots/eval_price_${counter}.png
	((counter+=50))
done
echo All done