"""Offline wrapper checks. Fixtures never use /data, real credentials or network."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import run_receipt_recovery as runner


class WrapperHelpers(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='receipt-wrapper-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.root.chmod(0o700)

    def test_log_writes_private_new_file_and_keeps_descriptor(self):
        path = self.root / 'events.jsonl'
        log = runner.EventLog(path)
        self.addCleanup(log.close)
        fd = log.fd
        log.append({'one': 1}); log.append({'two': 2})
        self.assertEqual(log.fd, fd)
        self.assertEqual([json.loads(x) for x in path.read_text().splitlines()], [{'one': 1}, {'two': 2}])
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_log_refuses_existing_file_without_append(self):
        path = self.root / 'events.jsonl'; path.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError): runner.EventLog(path)
        self.assertEqual(path.read_bytes(), b'preserve')

    def test_log_refuses_symlink_and_symlinked_parent(self):
        target = self.root / 'target'; target.write_bytes(b'preserve')
        path = self.root / 'events'; path.symlink_to(target)
        with self.assertRaises(OSError): runner.EventLog(path)
        link = self.root / 'parent-link'; link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError): runner.EventLog(link / 'new-events')
        self.assertEqual(target.read_bytes(), b'preserve')

    def test_log_refuses_nonprivate_parent(self):
        sub = self.root / 'public'; sub.mkdir(mode=0o755)
        with self.assertRaises(ValueError): runner.EventLog(sub / 'events')
        self.assertFalse((sub / 'events').exists())

    def test_log_detects_hardlink_before_append(self):
        path = self.root / 'events'; log = runner.EventLog(path); self.addCleanup(log.close)
        os.link(path, self.root / 'second-name')
        with self.assertRaises(ValueError): log.append({'must_not_write': True})
        self.assertEqual(path.read_bytes(), b'')

    def test_log_detects_inode_substitution_before_append(self):
        path = self.root / 'events'; log = runner.EventLog(path); self.addCleanup(log.close)
        saved = self.root / 'original'; path.rename(saved); path.write_bytes(b'replacement')
        with self.assertRaises(ValueError): log.append({'must_not_write': True})
        self.assertEqual(saved.read_bytes(), b''); self.assertEqual(path.read_bytes(), b'replacement')

    def test_log_detects_parent_substitution(self):
        parent = self.root / 'private'; parent.mkdir(mode=0o700)
        log = runner.EventLog(parent / 'events'); self.addCleanup(log.close)
        saved = self.root / 'saved'; parent.rename(saved); parent.mkdir(mode=0o700)
        with self.assertRaises(ValueError): log.append({'must_not_write': True})
        self.assertEqual((saved / 'events').read_bytes(), b'')

    def test_bounded_file_rejects_symlink_hardlink_and_oversize(self):
        p = self.root / 'input'; p.write_bytes(b'abc'); p.chmod(0o600)
        self.assertEqual(runner.bounded_file(p, 3, private=True), b'abc')
        with self.assertRaises(ValueError): runner.bounded_file(p, 2)
        s = self.root / 'symlink'; s.symlink_to(p)
        with self.assertRaises(ValueError): runner.bounded_file(s, 3)
        os.link(p, self.root / 'hardlink')
        with self.assertRaises(ValueError): runner.bounded_file(p, 3)

    def test_pair_lock_refuses_symlink_and_hardlink(self):
        p = self.root / 'lock'; p.write_bytes(b'')
        s = self.root / 'symlink'; s.symlink_to(p)
        with self.assertRaises(OSError): runner.held_pair_lock(s)
        os.link(p, self.root / 'hardlink')
        with self.assertRaises(ValueError): runner.held_pair_lock(p)

    def test_pair_lock_detects_inode_substitution_during_open(self):
        p = self.root / 'lock'; p.write_bytes(b'')
        original_open = runner.os.open
        def replace_after_open(path, *args, **kwargs):
            fd = original_open(path, *args, **kwargs)
            if Path(path) == p:
                p.rename(self.root / 'old-lock'); p.write_bytes(b'')
            return fd
        with mock.patch.object(runner.os, 'open', side_effect=replace_after_open):
            with self.assertRaises(ValueError): runner.held_pair_lock(p)

    def test_pair_lock_contention(self):
        p = self.root / 'lock'; p.write_bytes(b'')
        first = runner.held_pair_lock(p)
        try:
            with self.assertRaises(OSError): runner.held_pair_lock(p)
        finally: first.close()
        with runner.held_pair_lock(p): pass




class WrapperIntegration(unittest.TestCase):
    """Real SQLite authority/controller/SafeDrive; synthetic HTTP and temp paths."""
    def setUp(self):
        import sys
        here = Path(__file__).resolve().parent
        if (here / 'gcs_drive_retained_recovery_authority.py').is_file():
            self.ops = here
            sys.path.insert(0, str(self.ops))
            self.addCleanup(lambda: sys.path.remove(str(self.ops)))
        else:
            canonical = Path('/Users/joshlacalamito/Developer/Catbird+Petrel/catbird-intel/ops/gce')
            self.ops = canonical if (canonical / 'gcs_drive_retained_recovery_authority.py').is_file() else Path('/Users/joshlacalamito/Documents/Codex/2026-10-21/task/diagnostic_v3_validation/catbird-intel/ops/gce')
            sys.path.insert(0, str(self.ops))
            self.addCleanup(lambda: sys.path.remove(str(self.ops)))
        from test_retained_transfer_authority import RetainedAuthorityTests

        self.fixture = RetainedAuthorityTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = Path(self.fixture.temp.name).resolve()

    def exercise(self, expired=False, payload_tail=False):
        import hashlib
        import sqlite3
        import sys
        import types
        import urllib.parse
        import urllib.error
        import time
        import gcs_drive_retained_recovery_authority as recovery
        import gcs_drive_retained_continuous as continuous
        import gcs_drive_retained_receipt_recovery as controller_module
        import gcs_drive_drain as drain
        import gce_segment_offloader as offloader
        import receipt_recovery_live_verifier as live
        from test_receipt_recovery_live_verifier import Clock, HTTP, Response, Auth, digests
        from gcs_drive_retained_overflow import pair_key
        fixture = self.fixture
        old, manifest, manifest_sha = fixture.open_run(1)
        fixture.reserve(old, manifest, manifest_sha)
        payload, receipt = b'tiny-payload-1', b'tiny-receipt-1'
        session = 'https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable&upload_id=synthetic-old'
        target = hashlib.sha256(session.encode()).hexdigest()
        prefix_len = 8
        tail_bytes = payload[prefix_len:]
        charged_body = tail_bytes if payload_tail else receipt
        content_range = (f'bytes {prefix_len}-{len(payload)-1}/{len(payload)}'
                         if payload_tail else f'bytes 0-{len(receipt)-1}/{len(receipt)}')
        operation = recovery._operation_key('PUT', 'media_chunk', target,
                                            content_range, charged_body)
        attempt, slot = old.start_request(len(charged_body), 20, 'media_chunk', method='PUT',
                                         target_category='media_chunk', target_sha256=target, operation_key=operation)
        fixture.clock.value = max(fixture.clock.value, slot)
        old.mark_dispatch(attempt); old.finish_request(attempt, 500)
        old.db.execute('UPDATE retained_account SET process_pid=? WHERE id=1', (99999999,)); old.db.commit()
        old.close()
        clock = Clock(); clock.wall = fixture.clock.value + 30; clock.tick = fixture.monotonic.value + 30
        key = pair_key(manifest); run_id = manifest['release_id']
        data = self.root / 'data'; data.mkdir(mode=0o700)
        work = data / ('gcs-drive-retained-' + run_id); stage = work / 'stage' / key; stage.mkdir(parents=True)
        names = {'payload':'seg_0000000001.jss','receipt':'seg_0000000001.jss.receipt.json'}
        bodies = {'payload':payload,'receipt':receipt}; pins = {k:digests(v) for k,v in bodies.items()}
        ids = {'folder':'folderSynthetic0123','payload':'payloadSynthetic0123','receipt':'receiptSynthetic0123'}
        entry = {'key':key,'status':'file_upload_inflight','release_id':run_id,'manifest_sha256':manifest_sha,
                 'archive_qualified':False,'drive':{k:{'id':v} for k,v in ids.items()}}
        if payload_tail:
            entry['drive']['payload']['session']=session
        else:
            entry['drive']['receipt']['session']=session
        for kind in bodies:
            (stage / names[kind]).write_bytes(bodies[kind])
            entry[kind+'_name']=names[kind];entry[kind+'_digest']=pins[kind]
            ref=manifest[kind];entry[kind+'_ref']={'name':ref['object'],'generation':ref['generation'],'size':ref['bytes']}
        journal = {'schema':'catbird.gcs-drive-drain.v1','bucket':manifest['bucket'],'root':manifest['drive_root'],
                   'source_id':manifest['source_id'],'source_epoch':manifest['source_epoch'],'segments':{key:entry},'folders':{}}
        journal_path=work/'journal.json';journal_raw=json.dumps(journal,sort_keys=True).encode();journal_path.write_bytes(journal_raw);journal_path.chmod(0o600)
        (work/'overflow.lock').write_bytes(b'');(work/'overflow.lock').chmod(0o600)
        buffer=data/'gcs-buffer';buffer.mkdir();(buffer/'status.json').write_text(json.dumps({'reason':'ready','updated_at_unix':clock.wall}))
        external=self.root/'external.json';external.write_text(json.dumps({'observed_at':clock.wall,'human_no_other_uploads':'No other uploads',
          'local_writer_processes':[],'netcup_writer_processes':[],'netcup_writer_timers':[],'netcup_writer_units':[]}));external.chmod(0o600)
        events=self.root/'events.jsonl'
        http,gcs,auth=HTTP(clock),HTTP(clock),Auth()
        def jsonroute(transport,url,value): transport.routes[HTTP.key(url)]=lambda:Response(json.dumps(value).encode())
        drivebase='https://www.googleapis.com/drive/v3/'
        jsonroute(http,drivebase+'about',{'user':{'permissionId':fixture.account_id}})
        jsonroute(http,drivebase+'files/'+manifest['drive_root'],{'id':manifest['drive_root'],'mimeType':live.FOLDER_MIME,'trashed':False})
        jsonroute(http,drivebase+'files/'+ids['folder'],{'id':ids['folder'],'name':manifest['folder_name'],'parents':[manifest['drive_root']],'mimeType':live.FOLDER_MIME,'trashed':False})
        state={'receipt_exists':not (expired or payload_tail)}
        for kind in bodies:
            ref=manifest[kind];name=names[kind];file_id=ids[kind]
            meta={'id':file_id,'name':name,'parents':[ids['folder']],'trashed':False,'mimeType':'application/octet-stream' if kind=='payload' else 'application/json','size':str(ref['bytes']),'md5Checksum':pins[kind]['md5']}
            http.routes[HTTP.key(drivebase+'files/'+file_id)] = lambda kind=kind,meta=meta: Response(json.dumps(meta).encode()) if kind!='receipt' or state['receipt_exists'] else Response(b'{}',status=404)
            http.routes[HTTP.key(drivebase+'files/'+file_id+'?alt=media')] = lambda kind=kind:Response(bodies[kind],headers={'Content-Length':str(len(bodies[kind]))})
            import base64
            gsbase='https://storage.googleapis.com/storage/v1/b/'+manifest['bucket']+'/o/'+urllib.parse.quote(ref['object'],safe='')
            jsonroute(gcs,gsbase,{'bucket':manifest['bucket'],'name':ref['object'],'generation':str(ref['generation']),'size':str(ref['bytes']),
                       'md5Hash':base64.b64encode(bytes.fromhex(pins[kind]['md5'])).decode(),'metadata':{'sha256':ref['sha256']}})
            mediaurl=gsbase.replace('/storage/v1/','/download/storage/v1/')+'?alt=media'
            gcs.routes[HTTP.key(mediaurl)]=lambda kind=kind,ref=ref:Response(bodies[kind],headers={'Content-Length':str(ref['bytes']),'x-goog-generation':str(ref['generation'])})
        original_live=live.ReceiptRecoveryLiveVerifier;verifiers=[]
        def make_live(**kwargs):
            v=original_live(**kwargs,gcs_http=gcs,drive_http=http,drive_auth=auth,clock=clock.time,mono=clock.monotonic,sleep=clock.sleep);verifiers.append(v);return v
        requests=[]
        class UploadResponse(io.BytesIO):
            def __init__(self,body,status=200,headers=None): super().__init__(body);self.status=status;self.headers=headers or {}
        class Opener:
            def open(inner, req, timeout):
                requests.append((req.get_method(),req.full_url,req.data,dict(req.header_items())))
                if payload_tail:
                    if len(requests)==1:
                        raise urllib.error.HTTPError(req.full_url,308,'resume',{'Range':f'bytes=0-{prefix_len-1}'},io.BytesIO(b''))
                    if len(requests)==2:
                        self.assertEqual(req.data,tail_bytes)
                        return UploadResponse(json.dumps({'id':ids['payload']}).encode())
                    if req.get_method()=='POST':
                        return UploadResponse(b'',headers={'Location':session.replace('synthetic-old','synthetic-new')})
                    self.assertEqual(req.data,receipt);state['receipt_exists']=True
                    return UploadResponse(json.dumps({'id':ids['receipt']}).encode())
                if len(requests)==1:
                    if expired:
                        code = 404 if expired is True else int(expired)
                        raise urllib.error.HTTPError(req.full_url,code,'synthetic expired',{},io.BytesIO(b''))
                    return UploadResponse(json.dumps({'id':ids['receipt']}).encode())
                if req.get_method()=='POST':
                    return UploadResponse(b'',headers={'Location':session.replace('synthetic-old','synthetic-new')})
                self.assertEqual(req.data,receipt);state['receipt_exists']=True
                return UploadResponse(json.dumps({'id':ids['receipt']}).encode())
        real_path=Path
        def mapped_path(value):
            p=real_path(value)
            if p==real_path('/data') or str(p).startswith('/data/'):return data/str(p)[6:]
            if str(p).startswith('/proc/'):return self.root/'absent-process'
            return p
        original_require=runner.require
        def require_nonroot(ok,label):
            if label!='root_owner_required': original_require(ok,label)
        original_controller=controller_module.ReceiptRecoveryController
        class ClockedController(original_controller):
            def __init__(self, **kwargs): super().__init__(**kwargs,clock=clock.time)
        original_inspect=recovery.RetainedObservationAuthority.inspect_account
        def inspect_account(*args,**kwargs):
            kwargs.setdefault('clock',clock.time);return original_inspect(*args,**kwargs)
        original_open_new=recovery.RecoveryAuthority.open_new
        def open_new(*args,**kwargs):return original_open_new(*args,**kwargs,clock=clock.time,monotonic_clock=clock.monotonic)
        argv=['run_receipt_recovery.py','--runtime',str(self.ops),'--run-id',run_id,'--attempt-id',attempt,
              '--manifest-sha256',manifest_sha,'--journal-sha256',hashlib.sha256(journal_raw).hexdigest(),
              '--external-facts',str(external),'--external-sha256',hashlib.sha256(external.read_bytes()).hexdigest(),
              '--events',str(events),'--deadline-epoch',str(clock.wall+600),'--execute-receipt-recovery']
        output=io.StringIO(); caught=[]
        def trace(frame,event,arg):
            if event == "exception" and type(arg[1]).__name__ in ("Conflict","AuthorityBlocked"):
                caught.append((frame.f_code.co_name,str(arg[1])))
            return trace
        with contextlib.ExitStack() as stack:
            patches=[mock.patch.object(runner,'DB',fixture.db_path),mock.patch.object(runner,'ACCOUNT',fixture.account_id),mock.patch.object(runner,'WRITER',fixture.writer_id),
                     mock.patch.object(runner,'Path',side_effect=mapped_path),mock.patch.object(runner,'require',side_effect=require_nonroot),
                     mock.patch.object(runner.time,'time',side_effect=clock.time),mock.patch.object(runner.time,'sleep',side_effect=clock.sleep),mock.patch.object(runner.time,'monotonic',side_effect=clock.monotonic),
                     mock.patch.object(runner.shutil,'disk_usage',return_value=types.SimpleNamespace(free=200<<30)),
                     mock.patch.object(runner.subprocess,'run',return_value=types.SimpleNamespace(returncode=0,stdout='inactive')),
                     mock.patch.object(continuous,'local_writer_preflight',return_value={'synthetic':True}),
                     mock.patch.object(recovery,'_boot_id',return_value='synthetic-boot'),mock.patch.object(controller_module,'ReceiptRecoveryController',ClockedController),mock.patch.object(recovery.RetainedObservationAuthority,'inspect_account',side_effect=inspect_account),mock.patch.object(recovery.RecoveryAuthority,'open_new',side_effect=open_new),
                     mock.patch.object(live,'ReceiptRecoveryLiveVerifier',side_effect=make_live),mock.patch.object(offloader,'DriveClient',return_value=Auth()),
                     mock.patch.object(drain,'_NO_REDIRECT_OPENER',Opener()),mock.patch.object(sys,'argv',argv)]
            for patch in patches:stack.enter_context(patch)
            stack.enter_context(contextlib.redirect_stdout(output))
            previous_trace=sys.gettrace();sys.settrace(trace)
            try: code=runner.main()
            finally: sys.settrace(previous_trace)
        records=[json.loads(line) for line in events.read_text().splitlines()]
        self.assertEqual(code,0,(records,caught[-8:]))
        with contextlib.closing(sqlite3.connect(fixture.db_path)) as db:
            db.row_factory=sqlite3.Row
            original=dict(db.execute('SELECT * FROM retained_attempt WHERE id=?',(attempt,)).fetchone())
            self.assertEqual((original['http_status'],original['status'],original['bytes']),(500,'unresolved',len(charged_body)))
            self.assertEqual(db.execute('SELECT count(*) FROM retained_attempt').fetchone()[0],1)
            self.assertEqual(db.execute('SELECT count(*) FROM retained_receipt_recovery_attempt').fetchone()[0],len(requests))
        expected_reqs = 4 if payload_tail else (3 if expired else 1)
        expected_put_bytes = (len(tail_bytes) + len(receipt)) if payload_tail else (len(receipt) if expired else 0)
        self.assertEqual(len(requests),expected_reqs)
        self.assertEqual(sum(len(r[2] or b'') for r in requests if r[0]=='PUT'),expected_put_bytes)
        self.assertEqual(len(gcs.calls),4)
        self.assertEqual(len(http.calls),18 if (expired or payload_tail) else 11)
        self.assertNotIn(session,output.getvalue());self.assertNotIn('synthetic-token',output.getvalue())
        self.assertTrue(all((stage/n).exists() for n in names.values()))
        self.assertTrue(len(verifiers) == 1 and verifiers[0]._closed)

    def test_full_wrapper_completed_probe_real_ledger_and_readbacks(self):self.exercise()
    def test_full_wrapper_expired_same_id_real_ledger_and_readbacks(self):self.exercise(expired=True)
    def test_full_wrapper_expired_410_same_id_real_ledger_and_readbacks(self):self.exercise(expired=410)
    def test_full_wrapper_payload_tail_and_receipt_auto_real_ledger_and_readbacks(self):self.exercise(payload_tail=True)

if __name__ == '__main__': unittest.main()


