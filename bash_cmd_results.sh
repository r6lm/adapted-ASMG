# Incremental Update
printf "#######################################################################\nIU results:\n#"
cat MovieLens/IU/ckpts/IU_train11-23_test24-30_1epoch_0.001/period30/test_metrics.txt
printf "\n"

# SMLmf
printf "#######################################################################\nSMLmf results:\n#"
cat MovieLens/SMLmf/SML/ckpts/SMLmf_train11-23_test24-30_1epoch_1epoch_0.01_0.01/period30/test_metrics.txt
printf "\n"


