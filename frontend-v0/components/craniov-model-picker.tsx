'use client'

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'

interface ModelMetrics {
  name: string
  dice: string
  volume: string
}

export function ModelPickerTabs() {
  const models: ModelMetrics[] = [
    { name: 'Attention U-Net', dice: '0.892', volume: '58.3 cm³' },
    { name: 'SwinUNETR', dice: '0.901', volume: '59.1 cm³' },
    { name: 'nnU-Net', dice: '0.908', volume: '58.7 cm³' },
    { name: 'Ensemble', dice: '0.905', volume: '58.7 cm³' },
  ]

  return (
    <Card className="border border-gray-200 bg-white">
      <CardHeader>
        <CardTitle>Choose Prediction</CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="Ensemble" className="w-full">
          <TabsList className="grid w-full grid-cols-4 bg-gray-100">
            {models.map((model) => (
              <TabsTrigger key={model.name} value={model.name} className="text-xs sm:text-sm">
                {model.name}
              </TabsTrigger>
            ))}
          </TabsList>
          {models.map((model) => (
            <TabsContent key={model.name} value={model.name} className="mt-4 space-y-3">
              <div className="flex gap-4">
                <Badge variant="secondary" className="bg-gray-100">
                  Mean Dice: {model.dice}
                </Badge>
                <Badge variant="secondary" className="bg-gray-100">
                  Volume: {model.volume}
                </Badge>
              </div>
            </TabsContent>
          ))}
        </Tabs>
      </CardContent>
    </Card>
  )
}
