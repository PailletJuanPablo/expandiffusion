import { Studio } from './components/Studio'
import { TooltipProvider } from './components/ui/tooltip'

export default function App() {
  return (
    <TooltipProvider delayDuration={350}>
      <Studio />
    </TooltipProvider>
  )
}
