(function(w,d,s,src,mid){
  var js = d.createElement(s);
  js.async = true;
  js.src = src + '?id=' + mid;
  d.head.appendChild(js);
  w.dataLayer = w.dataLayer || [];
  function gtag(){w.dataLayer.push(arguments);}
  w.gtag = gtag;
  gtag('js', new Date());
  gtag('config', mid, { debug_mode: true }); // Optional: view events in GA4 DebugView
})(window, document, 'script', 'https://www.googletagmanager.com/gtag/js', 'G-71BJZLVZ2K');