# Patch for making SimSwap CPU-compatible.
# Should be executed from SimSwap folder.

<# :
@echo off
powershell -noprofile -NoLogo "iex (${%~f0} | out-string)"
exit /b
: #>
<# :
@echo off
powershell -noprofile -NoLogo "iex (${%~f0} | out-string)"
exit /b
: #>
$search_replace = @{
    "if len(self.opt.gpu_ids)" = "if torch.cuda.is_available() and len(self.opt.gpu_ids)";
    "torch.device(`"cuda:0`")" = "torch.device(`'cuda:0`' if torch.cuda.is_available() else `'cpu`')";
    "torch.load(netArc_checkpoint)" = "torch.load(netArc_checkpoint) if torch.cuda.is_available() else torch.load(netArc_checkpoint, map_location=torch.device(`'cpu`'))";
	"net.load_state_dict(torch.load(save_pth))" = "net.load_state_dict(torch.load(save_pth)) if torch.cuda.is_available() else net.load_state_dict(torch.load(save_pth, map_location=torch.device(`'cpu`')))";
    ".cuda()" = ".to(torch.device(`'cuda:0`' if torch.cuda.is_available() else `'cpu`'))";
    ".to('cuda')" = ".to(torch.device(`'cuda:0`' if torch.cuda.is_available() else `'cpu`'))";
}

ForEach ($File in (Get-ChildItem -Path '.\*.py' -Recurse -File)) {
    $content = (Get-Content $File)
    if ($content.length -gt 0) {
        ForEach ($search in $search_replace.Keys) {
            $content = $content.replace("$search", "$($search_replace[$search])")
        }
        Set-Content $File $content
    }
}
