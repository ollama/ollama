# PHASE 3: UI/UX İYİLEŞTİRMELERİ VE YENİ LAYOUT SİSTEMİ

## 📋 GENEL BAKIŞ
Modern, performanslı ve kullan

ıcı dostu bir arayüz tasarımı. Multi-panel layout, animasyonlar, glassmorphism efektleri ve responsive design.

## 🎯 HEDEFLER
1. ✅ Multi-panel layout (sidebar, main, inspector)
2. ✅ Sekmeler/tabs sistemi
3. ✅ Glassmorphism & blur efektleri
4. ✅ Smooth animasyonlar (Framer Motion)
5. ✅ Dark/Light theme toggle
6. ✅ Responsive design
7. ✅ Performance optimizasyonu
8. ✅ Keyboard shortcuts

---

## 🏗️ YENİ LAYOUT YAPISI

```
┌─────────────────────────────────────────────────────────────┐
│                        TOPBAR                               │
│  Logo | Tabs | Provider Selector | Theme | User            │
├────────┬─────────────────────────────────────┬──────────────┤
│        │                                     │              │
│ SIDE   │          MAIN PANEL                 │  INSPECTOR   │
│ BAR    │                                     │    PANEL     │
│        │  ┌──────────────────────────────┐  │              │
│ • Chat │  │                              │  │  Context     │
│ • New  │  │      Chat Messages           │  │  Stats       │
│ • ...  │  │                              │  │              │
│        │  │                              │  │  Cost        │
│        │  └──────────────────────────────┘  │  Tracker     │
│        │                                     │              │
│        │  ┌──────────────────────────────┐  │  Settings    │
│        │  │   Chat Input Form            │  │              │
│        │  └──────────────────────────────┘  │              │
│        │                                     │              │
├────────┴─────────────────────────────────────┴──────────────┤
│                      STATUSBAR                              │
│  Model Status | Token Count | Cost | Speed                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 DOSYA DEĞİŞİKLİKLERİ

### 1. ROOT LAYOUT COMPONENT

**Dosya:** `/home/user/ollama/app/ui/app/src/routes/__root.tsx` (GÜNCELLENECEK)

```typescript
import { Outlet, createRootRoute } from '@tanstack/react-router';
import { TanStackRouterDevtools } from '@tanstack/router-devtools';
import { Sidebar } from '../components/layout/Sidebar';
import { Topbar } from '../components/layout/Topbar';
import { Inspector } from '../components/layout/Inspector';
import { Statusbar } from '../components/layout/Statusbar';
import { useSettings } from '../hooks/useSettings';
import { AnimatePresence, motion } from 'framer-motion';

export const Route = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  const { data: settings } = useSettings();
  const sidebarOpen = settings?.sidebar_open ?? true;
  const inspectorOpen = settings?.inspector_open ?? true;

  return (
    <div className="flex flex-col h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      {/* Topbar */}
      <Topbar />

      {/* Main Content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Sidebar */}
        <AnimatePresence>
          {sidebarOpen && (
            <motion.div
              initial={{ x: -280, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: -280, opacity: 0 }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="w-70 border-r border-gray-200 dark:border-gray-700"
            >
              <Sidebar />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Main Panel */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <Outlet />
        </div>

        {/* Inspector Panel */}
        <AnimatePresence>
          {inspectorOpen && (
            <motion.div
              initial={{ x: 320, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: 320, opacity: 0 }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="w-80 border-l border-gray-200 dark:border-gray-700"
            >
              <Inspector />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Statusbar */}
      <Statusbar />

      {/* Dev Tools */}
      {import.meta.env.DEV && <TanStackRouterDevtools />}
    </div>
  );
}
```

---

### 2. GLASSMORPHISM STYLES

**Dosya:** `/home/user/ollama/app/ui/app/src/styles/glass.css` (YENİ)

```css
/* Glassmorphism Effects */
.glass {
  background: rgba(255, 255, 255, 0.7);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.3);
  box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
}

.glass-dark {
  background: rgba(17, 25, 40, 0.75);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.1);
  box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

.glass-strong {
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
}

/* Animated Gradient Background */
.gradient-bg {
  background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
  background-size: 400% 400%;
  animation: gradientShift 15s ease infinite;
}

@keyframes gradientShift {
  0% {
    background-position: 0% 50%;
  }
  50% {
    background-position: 100% 50%;
  }
  100% {
    background-position: 0% 50%;
  }
}

/* Smooth Transitions */
.smooth-transition {
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Hover Effects */
.hover-lift {
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.hover-lift:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 24px rgba(0, 0, 0, 0.1);
}

/* Focus Ring */
.focus-ring {
  @apply focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 dark:focus:ring-offset-gray-900;
}
```

---

### 3. TABS COMPONENT

**Dosya:** `/home/user/ollama/app/ui/app/src/components/Tabs.tsx` (YENİ)

```typescript
import { Tab } from '@headlessui/react';
import { motion } from 'framer-motion';
import { XMarkIcon } from '@heroicons/react/24/outline';

interface TabItem {
  id: string;
  title: string;
  closeable?: boolean;
}

interface TabsProps {
  tabs: TabItem[];
  activeTab: string;
  onTabChange: (tabId: string) => void;
  onTabClose?: (tabId: string) => void;
}

export function Tabs({ tabs, activeTab, onTabChange, onTabClose }: TabsProps) {
  const selectedIndex = tabs.findIndex(t => t.id === activeTab);

  return (
    <Tab.Group selectedIndex={selectedIndex} onChange={(index) => onTabChange(tabs[index].id)}>
      <Tab.List className="flex space-x-1 bg-gray-100 dark:bg-gray-800 p-1 rounded-lg">
        {tabs.map((tab) => (
          <Tab
            key={tab.id}
            className={({ selected }) =>
              `group relative flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-md focus:outline-none focus-ring transition-all ${
                selected
                  ? 'bg-white dark:bg-gray-700 text-indigo-600 dark:text-indigo-400 shadow'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
              }`
            }
          >
            {({ selected }) => (
              <>
                <span>{tab.title}</span>
                {tab.closeable && onTabClose && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onTabClose(tab.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 p-0.5 rounded hover:bg-gray-300 dark:hover:bg-gray-600 transition-opacity"
                  >
                    <XMarkIcon className="h-4 w-4" />
                  </button>
                )}
                {selected && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute inset-0 bg-white dark:bg-gray-700 rounded-md -z-10"
                    transition={{ type: 'spring', bounce: 0.2, duration: 0.6 }}
                  />
                )}
              </>
            )}
          </Tab>
        ))}
      </Tab.List>
    </Tab.Group>
  );
}
```

---

### 4. KEYBOARD SHORTCUTS

**Dosya:** `/home/user/ollama/app/ui/app/src/hooks/useKeyboardShortcuts.ts` (YENİ)

```typescript
import { useEffect } from 'react';

interface ShortcutConfig {
  key: string;
  ctrl?: boolean;
  shift?: boolean;
  alt?: boolean;
  callback: () => void;
  description: string;
}

export function useKeyboardShortcuts(shortcuts: ShortcutConfig[]) {
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      for (const shortcut of shortcuts) {
        const ctrlMatch = shortcut.ctrl ? e.ctrlKey || e.metaKey : !e.ctrlKey && !e.metaKey;
        const shiftMatch = shortcut.shift ? e.shiftKey : !e.shiftKey;
        const altMatch = shortcut.alt ? e.altKey : !e.altKey;

        if (
          e.key.toLowerCase() === shortcut.key.toLowerCase() &&
          ctrlMatch &&
          shiftMatch &&
          altMatch
        ) {
          e.preventDefault();
          shortcut.callback();
          break;
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [shortcuts]);
}

// Predefined shortcuts
export const SHORTCUTS: ShortcutConfig[] = [
  {
    key: 'k',
    ctrl: true,
    description: 'Focus search / command palette',
    callback: () => {},
  },
  {
    key: 'n',
    ctrl: true,
    description: 'New chat',
    callback: () => {},
  },
  {
    key: 'b',
    ctrl: true,
    description: 'Toggle sidebar',
    callback: () => {},
  },
  {
    key: 'i',
    ctrl: true,
    description: 'Toggle inspector',
    callback: () => {},
  },
  {
    key: '/',
    ctrl: true,
    description: 'Show shortcuts',
    callback: () => {},
  },
];
```

---

## 📊 PERFORMANS KRİTERLERİ

- **Layout Render:** < 16ms (60fps)
- **Animation Frame:** 60fps consistently
- **Blur Effect:** Hardware accelerated
- **Tab Switch:** < 100ms
- **Sidebar Toggle:** < 200ms

---

## ✅ BAŞARI KRİTERLERİ

1. ✅ Multi-panel layout çalışıyor
2. ✅ Tabs smooth animasyonlar ile açılıyor/kapanıyor
3. ✅ Glassmorphism efektleri aktif
4. ✅ Keyboard shortcuts çalışıyor
5. ✅ Responsive design tüm ekran boyutlarında
6. ✅ 60fps sabit animasyon

**SONRAKİ PHASE:** Phase 4 - Advanced Chat Features
