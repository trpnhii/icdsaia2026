import re, json, sys
s=open('data.js','r',encoding='utf-8').read()
idx=s.find('window.DASHBOARD_DATA')
if idx==-1:
    print('no assignment')
    sys.exit(1)
start = s.find('{', idx)
end = s.rfind('};')
obj_text = s[start:end+1]
# remove trailing commas
obj_text = re.sub(r',\s*(\}|\])', r"\1", obj_text)
# remove JS comments
obj_text = re.sub(r'//.*', '', obj_text)
# load
try:
    data=json.loads(obj_text)
except Exception as e:
    print('json load error',e)
    sys.exit(1)
fin=data.get('financial',{})
print('financial irr_monthly:', fin.get('irr_monthly'))
print('capex_ai_vnd:', fin.get('capex_ai_vnd'))

timeline=data.get('timeline',[])
cashflows=[-(fin.get('capex_ai_vnd') or 0.0)]
for row in timeline:
    cashflows.append(row.get('incremental_cashflow_vnd',0.0))
print('cashflows preview:', cashflows[:8])

# IRR function
from math import isfinite

def irr(cashflows):
    if not any(cf>0 for cf in cashflows) or not any(cf<0 for cf in cashflows):
        return None
    def npv_at(r):
        return sum(cf/((1+r)**i) for i,cf in enumerate(cashflows))
    low, high = -0.999999, 10.0
    if npv_at(low)*npv_at(high) > 0:
        print('npv_at_low, npv_at_high:', npv_at(low), npv_at(high))
        return None
    for _ in range(200):
        mid=(low+high)/2
        midv=npv_at(mid)
        if abs(midv)<1e-7:
            return mid
        if npv_at(low)*midv<=0:
            high=mid
        else:
            low=mid
    return (low+high)/2

print('computed irr:', irr(cashflows))
