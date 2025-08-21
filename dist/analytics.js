(function(w,d,s,src,mid){
  var js = d.createElement(s);
  js.async = true;
  js.src = src + '?id=' + mid;
  d.head.appendChild(js);
  w.dataLayer = w.dataLayer || [];
  function gtag(){w.dataLayer.push(arguments);}
  w.gtag = gtag;
  gtag('js', new Date());
  // Disable auto page_view; we send page_view manually on first interaction.
  gtag('config', mid, { send_page_view: false, debug_mode: true });
})(window, document, 'script', 'https://www.googletagmanager.com/gtag/js', 'G-71BJZLVZ2K');