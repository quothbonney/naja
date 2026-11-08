"""Independent11UTC hold cleanup and active-controller restart safeguard."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

B=Path(__file__).resolve().parents[1]
A=B/'audit/high-throughput'
lock=(A/'deadline-guard.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(A/'deadline-guard.pid').write_text(str(os.getpid())+'\n')
end=datetime(2026,11,8,11,tzinfo=timezone.utc).timestamp()
last_restart=0
status_path=A/'deadline-guard-status.json'

def read(path,default):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default

def log(*values):
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),*values,flush=True)

def controller_alive():
    try:
        pid=int((A/'full64-controller.pid').read_text())
        proc=Path(f'/proc/{pid}')
        return proc.stat().st_uid==os.getuid() and 'full64_sequence_controller.py' in (proc/'cmdline').read_text()
    except (OSError,ValueError):return False

def info(job):
    result=subprocess.run(['scontrol','show','job',job,'-o'],capture_output=True,text=True)
    data=dict(re.findall(r'(\w+)=([^ ]*)',result.stdout)) if result.returncode==0 else {}
    return data if data.get('WorkDir') in (str(B),str(B.parent)) and f'({os.getuid()})' in data.get('UserId','') else {}

def cleanup(deadline):
    held=read(A/'full64-held-jobs.json',{})
    state=read(A/'full64-state.json',{})
    actions=[]
    for job,row in held.items():
        if row.get('status') not in ('held','restore_failed'):continue
        data=info(job)
        if data.get('JobState')!='PENDING':continue
        command=['scancel',job] if deadline else ['scontrol','release',job]
        result=subprocess.run(command,capture_output=True,text=True)
        actions.append(dict(job=job,command=command,returncode=result.returncode,stderr=result.stderr.strip()))
    for attempt in state.get('attempts',[]):
        for key in ('sample_job','metrics_job'):
            job=attempt.get(key)
            if not job:continue
            data=info(job)
            if data.get('JobState') not in ('PENDING','RUNNING','CONFIGURING','COMPLETING'):continue
            result=subprocess.run(['scancel',job],capture_output=True,text=True)
            actions.append(dict(job=job,command=['scancel',job],returncode=result.returncode,stderr=result.stderr.strip()))
    return actions

while time.time()<end:
    state=read(A/'full64-state.json',{})
    alive=controller_alive()
    marker=dict(checked_utc=datetime.now(timezone.utc).isoformat(),controller_alive=alive,sequence_status=state.get('status'),hard_deadline_utc='2026-11-08T11:00:00Z')
    if state.get('status')=='active' and not alive and time.time()-last_restart>300:
        log('Restarting unexpectedly absent active full64 sequencer')
        with (A/'full64-controller.log').open('a') as output:
            proc=subprocess.Popen([sys.executable,'-u',str(B/'scripts/full64_sequence_controller.py')],cwd=B.parent,
                stdin=subprocess.DEVNULL,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        last_restart=time.time();marker['restarted_pid']=proc.pid
    elif state.get('status') not in (None,'active'):
        marker['terminal_cleanup_actions']=cleanup(False)
    temp=status_path.with_suffix('.tmp');temp.write_text(json.dumps(marker,indent=2)+'\n');os.replace(temp,status_path)
    time.sleep(min(30,max(0,end-time.time())))
actions=cleanup(True)
(A/'deadline-cleanup.json').write_text(json.dumps(dict(deadline_utc='2026-11-08T11:00:00Z',actions=actions),indent=2)+'\n')
log('Deadline cleanup complete',actions)
