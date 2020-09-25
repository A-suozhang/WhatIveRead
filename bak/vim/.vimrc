set nocompatible   " 必须, 关闭 vi 兼容模式
filetype off       " 必须
 
" 设置 Runtime Path，供 Vundle 初始化
set rtp+=~/.vim/bundle/Vundle.vim
call vundle#begin()
 
" 让 Vundle 管理 Vundle，必须
Plugin 'gmarik/Vundle.vim'
Plugin 'rking/ag.vim'
Plugin 'kien/ctrlp.vim'
Plugin 'Yggdroot/indentLine'
Plugin 'scrooloose/syntastic'
Plugin 'scrooloose/nerdtree'
Plugin 'majutsushi/tagbar'
Plugin 'bling/vim-airline'
Plugin 'jrosiek/vim-mark'
Plugin 'bigeagle/molokai'
Plugin 'aceofall/gtags.vim'
Plugin 'hdima/python-syntax'
Plugin 'hynek/vim-python-pep8-indent'
 
call vundle#end()
 
colorscheme molokai " 使用 molokai 主题，请注意需要 256 色终端
set so=10     " 光标到倒数第10行开始滚屏   
syntax on     " 语法高亮
set number    " 显示行号
set autochdir " 打开文件时，自动 cd 到文件所在目录
set hlsearch 

" 文件类型支持
filetype on
filetype plugin on
filetype indent on
 
set list lcs=tab:\¦\   " 使用 ¦ 来显示 tab 缩进
 
" 缩进相关
set shiftwidth=4
set tabstop=4
set softtabstop=4
set smartindent
 
if has("autocmd")  " 打开时光标放在上次退出时的位置
    autocmd BufReadPost *
        \ if line("'\"") > 0 && line ("'\"") <= line("$") |
        \   exe "normal g'\"" |
        \ endif
endif
 
set completeopt=longest,menu " 自动补全菜单
 
" 鼠标支持
if has('mouse')
    set mouse=a
    set selectmode=mouse,key
    set nomousehide
endif
 
set modeline      " 底部的模式行
set cursorline    " 高亮光标所在行
" set cursorcolumn  " 高亮光标所在列
 
set showmatch     " 高亮括号匹配
set matchtime=0
set nobackup      " 关闭自动备份
 
set backspace=indent,eol,start
 
 
" 文件编码
set fenc=utf-8
set fencs=utf-8,gbk,gb18030,gb2312,cp936,usc-bom,euc-jp
set enc=utf-8
 
" 语法折叠
set foldmethod=syntax
set foldcolumn=0  " 设置折叠区域的宽度
set foldlevel=100
" 用空格键来开关折叠
nnoremap <space> @=((foldclosed(line('.')) < 0) ? 'zc' : 'zo')<CR>
 
 
set smartcase
set ignorecase  " 搜索时，智能大小写
set nohlsearch  " 关闭搜索高亮
set incsearch   " 实时显示搜索结果
 
 
" 让 j, k 可以在 自动wrap的行中上下移动
vmap j gj
vmap k gk
nmap j gj
nmap k gk
 
" Shift-T 开新 Tab
nmap T :tabnew<cr>
 
" 以下文件类型，敲 {<回车> 后，自动加入反括号 }
au FileType c,cpp,h,java,css,js,nginx,scala,go inoremap  <buffer>  {<CR> {<CR>}<Esc>O
 
" ------ python  ------------
let python_highlight_all = 1
au FileType python setlocal ts=4 sts=4 sw=4 smarttab expandtab
" ------ python end ---------
 
" ------- Tagbar ------------------
let g:tagbar_width = 30
nmap tb :TagbarToggle<cr>	" 使用 tb 开/关 Tagbar
 
" ------- Tagbar End --------------
 
" ------- Gtags -------------------
set cscopetag				 " 使用 cscope 作为 tags 命令
set cscopeprg='gtags-cscope' " 使用 gtags-cscope 代替 cscope
let GtagsCscope_Auto_Load = 1
let GtagsCscope_Quiet = 1
 
" ------- Gtags End ---------------
 
" ------- NerdTree -------------------
 
nmap nt :NERDTreeToggle<cr>
let NERDTreeShowBookmarks=0
let NERDTreeMouseMode=2
let g:nerdtree_tabs_focus_on_files=1
let g:nerdtree_tabs_open_on_gui_startup=0
 
let NERDTreeWinSize=25
let NERDTreeIgnore = ['\.pyc$']
let NERDTreeMinimalUI=0
let NERDTreeDirArrows=1
 
"let g:newrw_ftp_cmd = 'lftp'
let g:netrw_altv          = 1
let g:netrw_fastbrowse    = 2
let g:netrw_keepdir       = 1
let g:netrw_liststyle     = 3
let g:netrw_retmap        = 1
let g:netrw_silent        = 1
let g:netrw_special_syntax= 1
let g:netrw_browse_split = 3
let g:netrw_banner = 0
" ------- NerdTree End ---------------
 
" ---- Airline 
" set laststatus=2
" let g:airline#extensions#tabline#enabled = 0
" let g:airline_powerline_fonts = 0
" let g:airline_theme = "powerlineish"
" ---- Airline
 
 
" vim: ft=vim

" So I can move in insert mode
" inoremap <C-k> <C-o>gk
" inoremap <C-h> <Left>
" inoremap <C-l> <Right>
" inoremap <C-j> <C-o>gj

inoremap <C-o> <Esc>o
inoremap <C-j> <Down>
inoremap <C-h> <Left>
inoremap <C-l> <Right>
inoremap <C-k> <Up>
inoremap <C-a> <Esc>0wi
inoremap <C-e> <Esc>A

" move cursor into the middle of ()
imap () ()<Left>
imap [] []<Left>
imap {} {}<Left>
imap "" ""<Left>


" au BufWritePre * :set binary | set noeol
" au BufWritePost * :set nobinary | set eol
