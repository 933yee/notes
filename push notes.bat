@echo off
echo hexo clean begin..
call cmd /c "hexo clean"

echo hexo generate begin..
call cmd /c "hexo generate"

for /f "tokens=2 delims==" %%i in ('"wmic os get localdatetime /value"') do set datetime=%%i
set commit_date=%datetime:~0,4%/%datetime:~4,2%/%datetime:~6,2%

git add .
git commit -m "%commit_date%"
git subtree split --prefix=public -b gh-pages
git push origin gh-pages --force
git push origin master