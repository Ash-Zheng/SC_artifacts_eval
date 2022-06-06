# ================== table_length = 2500000 =================
# without Reordering
python single_table_breakdown.py --setting=1 --length=2500000 

# without intermediate result reuse
python single_table_breakdown.py --setting=2 --length=2500000 

# without in-advance gradient aggregation
python single_table_breakdown.py --setting=3 --length=2500000

# EL-Rec
python single_table_breakdown.py --setting=0 --length=2500000 


# ================== table_length = 5000000 =================
# without Reordering
python single_table_breakdown.py --setting=1 --length=5000000 

# without intermediate result reuse
python single_table_breakdown.py --setting=2 --length=5000000 

# without in-advance gradient aggregation
python single_table_breakdown.py --setting=3 --length=5000000

# EL-Rec
python single_table_breakdown.py --setting=0 --length=5000000 


# ================== table_length = 10000000 =================
# without Reordering
python single_table_breakdown.py --setting=1 --length=10000000 

# without intermediate result reuse
python single_table_breakdown.py --setting=2 --length=10000000 

# without in-advance gradient aggregation
python single_table_breakdown.py --setting=3 --length=10000000

# EL-Rec
python single_table_breakdown.py --setting=0 --length=10000000 