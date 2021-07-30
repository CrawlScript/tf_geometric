rm results.txt
#SCRIPT=bench_node_cls_early_stop_gat.py
#SCRIPT=bench_node_cls_early_stop_gcn.py
#SCRIPT=bench_node_cls_early_stop_appnp.py
SCRIPT=bench_node_cls_early_stop_sgc.py

for i in $(seq 1 20)
do
  python $SCRIPT
  python bench_report_results.py
done

