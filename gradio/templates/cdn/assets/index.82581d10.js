import{S as w,i as j,s as q,e as b,t as A,a as v,c as S,b as g,d as p,f as y,g as d,m as z,l as B,h as D,j as C,k,n as E,o as F,p as G,u as H,q as I,r as J}from"./index.98155427.js";import{C as K}from"./Column.2d03a686.js";function L(l){let e;const s=l[6].default,t=G(s,l,l[7],null);return{c(){t&&t.c()},m(n,i){t&&t.m(n,i),e=!0},p(n,i){t&&t.p&&(!e||i&128)&&H(t,s,n,n[7],e?J(s,n[7],i,null):I(n[7]),null)},i(n){e||(C(t,n),e=!0)},o(n){k(t,n),e=!1},d(n){t&&t.d(n)}}}function M(l){let e,s,t,n,i,u,m,r,f,_,o;return r=new K({props:{visible:l[3],$$slots:{default:[L]},$$scope:{ctx:l}}}),{c(){e=b("div"),s=b("div"),t=b("span"),n=A(l[0]),i=v(),u=b("span"),u.textContent="\u25BC",m=v(),S(r.$$.fragment),g(u,"class","transition"),p(u,"rotate-90",!l[3]),g(s,"class","w-full flex justify-between cursor-pointer"),g(e,"id",l[1]),g(e,"class","p-3 border border-gray-200 dark:border-gray-700 rounded-lg flex flex-col gap-3 hover:border-gray-300 dark:hover:border-gray-600 transition"),p(e,"hidden",!l[2])},m(a,c){y(a,e,c),d(e,s),d(s,t),d(t,n),d(s,i),d(s,u),d(e,m),z(r,e,null),f=!0,_||(o=B(s,"click",l[4]),_=!0)},p(a,[c]){(!f||c&1)&&D(n,a[0]),c&8&&p(u,"rotate-90",!a[3]);const h={};c&8&&(h.visible=a[3]),c&128&&(h.$$scope={dirty:c,ctx:a}),r.$set(h),(!f||c&2)&&g(e,"id",a[1]),c&4&&p(e,"hidden",!a[2])},i(a){f||(C(r.$$.fragment,a),f=!0)},o(a){k(r.$$.fragment,a),f=!1},d(a){a&&E(e),F(r),_=!1,o()}}}function N(l,e,s){let t,{$$slots:n={},$$scope:i}=e,{label:u}=e,{elem_id:m}=e,{visible:r=!0}=e,{open:f=!0}=e;const _=()=>{s(3,t=!t)};return l.$$set=o=>{"label"in o&&s(0,u=o.label),"elem_id"in o&&s(1,m=o.elem_id),"visible"in o&&s(2,r=o.visible),"open"in o&&s(5,f=o.open),"$$scope"in o&&s(7,i=o.$$scope)},l.$$.update=()=>{l.$$.dirty&32&&s(3,t=f)},[u,m,r,t,_,f,n,i]}class O extends w{constructor(e){super(),j(this,e,N,M,q,{label:0,elem_id:1,visible:2,open:5})}}var R=O;const T=["static"];export{R as Component,T as modes};
//# sourceMappingURL=index.82581d10.js.map